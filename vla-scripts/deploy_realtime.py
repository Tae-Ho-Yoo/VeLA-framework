"""
Two-loop CoT-VLA deployment server

Loop 1: Robot / VLA loop
    POST /act
    image + instruction + user_id
        -> read current execution speed from shared user profile
        -> OpenVLA predicts action trajectory
        -> return action + execution_speed
    IMPORTANT: This endpoint does NOT run CoT, so robot control is not blocked
    by the speed-adaptation reasoning loop.

Loop 2: Task-performance / CoT speed-adaptation loop
    POST /update_speed
    user_id + task_status from an external vision/task monitor
        -> compute personalized comfort zone
        -> fast policy handles clear cases immediately
        -> CoT is called only for ambiguous cases
        -> controller converts SpeedDecision + AdaptationLevel into target_speed
        -> save next current_speed to shared user profile

The robot trajectory is never modified by the speed adapter. Only execution_speed
is changed and applied separately in the UR5e execution layer.
"""

import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

json_numpy.patch()

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# ---------------------------------------------------------------------
# Speed / comfort-zone configuration
# ---------------------------------------------------------------------

DEFAULT_SPEED = 0.10
MIN_SPEED = 0.10
MAX_SPEED = 0.50

GLOBAL_COMFORT_MIN = 0.25
GLOBAL_COMFORT_MAX = 0.35
GLOBAL_COMFORT_CENTER = (GLOBAL_COMFORT_MIN + GLOBAL_COMFORT_MAX) / 2.0

SMALL_SPEED_STEP = 0.02
MEDIUM_SPEED_STEP = 0.05

MIN_SAMPLES_FOR_PERSONAL_ZONE = 2
MAX_HISTORY_PER_USER = 20
MIN_PERSONAL_ZONE_WIDTH = 0.02

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "user_speed_profiles.json"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clamp_speed(speed: float) -> float:
    return max(MIN_SPEED, min(MAX_SPEED, float(speed)))


def clamp_zone(zone_min: float, zone_max: float) -> tuple[float, float]:
    zone_min = clamp_speed(zone_min)
    zone_max = clamp_speed(zone_max)
    if zone_min > zone_max:
        zone_min, zone_max = zone_max, zone_min
    return zone_min, zone_max


def ensure_min_zone_width(center: float, zone_min: float, zone_max: float) -> tuple[float, float]:
    width = zone_max - zone_min
    if width >= MIN_PERSONAL_ZONE_WIDTH:
        return clamp_zone(zone_min, zone_max)

    half = MIN_PERSONAL_ZONE_WIDTH / 2.0
    return clamp_zone(center - half, center + half)


def make_default_profile() -> dict[str, Any]:
    # Keep the key name comfortable_speeds for backward compatibility.
    # Conceptually, these are manageable/successful speeds in the new design.
    return {
        "comfortable_speeds": [],
        "current_speed": None,
    }


def load_user_speed_profiles() -> dict[str, dict[str, Any]]:
    if not PROFILE_PATH.exists():
        return {}

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        cleaned: dict[str, dict[str, Any]] = {}
        for user_id, value in data.items():
            user_id = str(user_id)

            if isinstance(value, (int, float)):
                cleaned[user_id] = {
                    "comfortable_speeds": [clamp_speed(float(value))],
                    "current_speed": None,
                }
                continue

            if not isinstance(value, dict):
                cleaned[user_id] = make_default_profile()
                continue

            speeds = value.get("comfortable_speeds", [])
            if not isinstance(speeds, list):
                speeds = []

            safe_speeds = []
            for s in speeds:
                try:
                    safe_speeds.append(clamp_speed(float(s)))
                except (TypeError, ValueError):
                    continue

            current_speed = value.get("current_speed", None)
            try:
                current_speed = clamp_speed(float(current_speed)) if current_speed is not None else None
            except (TypeError, ValueError):
                current_speed = None

            cleaned[user_id] = {
                "comfortable_speeds": safe_speeds[-MAX_HISTORY_PER_USER:],
                "current_speed": current_speed,
            }

        return cleaned

    except Exception:
        logging.warning("[UserSpeedProfiles] Failed to load profile file. Starting empty.")
        return {}


def save_user_speed_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    safe_profiles: dict[str, dict[str, Any]] = {}

    for user_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue

        speeds = profile.get("comfortable_speeds", [])
        if not isinstance(speeds, list):
            speeds = []

        safe_speeds = []
        for s in speeds[-MAX_HISTORY_PER_USER:]:
            try:
                safe_speeds.append(round(clamp_speed(float(s)), 4))
            except (TypeError, ValueError):
                continue

        current_speed = profile.get("current_speed", None)
        try:
            current_speed = round(clamp_speed(float(current_speed)), 4) if current_speed is not None else None
        except (TypeError, ValueError):
            current_speed = None

        safe_profiles[str(user_id)] = {
            "comfortable_speeds": safe_speeds,
            "current_speed": current_speed,
        }

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(safe_profiles, f, indent=2, ensure_ascii=False)


def compute_personal_comfort_zone(
    user_id: Optional[str],
    profiles: dict[str, dict[str, Any]],
) -> tuple[float, float, float, bool, int]:
    if user_id is None or user_id not in profiles:
        return GLOBAL_COMFORT_MIN, GLOBAL_COMFORT_MAX, GLOBAL_COMFORT_CENTER, False, 0

    profile = profiles.get(user_id, {})
    speeds = profile.get("comfortable_speeds", [])
    if not isinstance(speeds, list):
        speeds = []

    safe_speeds = []
    for s in speeds:
        try:
            safe_speeds.append(clamp_speed(float(s)))
        except (TypeError, ValueError):
            continue

    num_samples = len(safe_speeds)
    if num_samples < MIN_SAMPLES_FOR_PERSONAL_ZONE:
        return GLOBAL_COMFORT_MIN, GLOBAL_COMFORT_MAX, GLOBAL_COMFORT_CENTER, False, num_samples

    zone_min = min(safe_speeds)
    zone_max = max(safe_speeds)
    center = (zone_min + zone_max) / 2.0
    zone_min, zone_max = ensure_min_zone_width(center, zone_min, zone_max)
    center = (zone_min + zone_max) / 2.0

    return zone_min, zone_max, center, True, num_samples


def get_base_speed(
    user_id: Optional[str],
    profiles: dict[str, dict[str, Any]],
) -> tuple[float, bool, str]:
    """
    Priority:
    1. current_speed
    2. most recently saved manageable/comfortable speed
    3. DEFAULT_SPEED
    """
    if user_id is None or user_id not in profiles:
        return DEFAULT_SPEED, False, "default"

    profile = profiles.get(user_id, {})

    current_speed = profile.get("current_speed", None)
    if current_speed is not None:
        try:
            return clamp_speed(float(current_speed)), True, "current_speed"
        except (TypeError, ValueError):
            pass

    speeds = profile.get("comfortable_speeds", [])
    if isinstance(speeds, list) and speeds:
        try:
            return clamp_speed(float(speeds[-1])), True, "manageable_speed"
        except (TypeError, ValueError):
            pass

    return DEFAULT_SPEED, False, "default"


def append_manageable_speed_to_profile(
    user_id: Optional[str],
    new_speed: Optional[float],
    profiles: dict[str, dict[str, Any]],
) -> bool:
    if user_id is None or new_speed is None:
        return False

    try:
        new_speed = clamp_speed(float(new_speed))
    except (TypeError, ValueError):
        return False

    if user_id not in profiles or not isinstance(profiles[user_id], dict):
        profiles[user_id] = make_default_profile()

    speeds = profiles[user_id].get("comfortable_speeds", [])
    if not isinstance(speeds, list):
        speeds = []

    speeds.append(new_speed)
    profiles[user_id]["comfortable_speeds"] = speeds[-MAX_HISTORY_PER_USER:]
    save_user_speed_profiles(profiles)
    return True


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return (
            f"{SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{instruction.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def sanitize_single_step_task(task: str) -> str:
    if not task:
        return ""

    lines = task.splitlines()
    task = lines[0].strip() if lines else ""
    if not task:
        return ""

    lower_task = task.lower()
    separators = [" and ", " then ", ", then ", ";"]
    for sep in separators:
        if sep in lower_task:
            idx = lower_task.find(sep)
            task = task[:idx].strip()
            lower_task = task.lower()
            break

    for word in ["slow", "slowly", "fast", "quickly", "gently", "rapidly"]:
        task = re.sub(rf"\b{word}\b", "", task, flags=re.IGNORECASE)

    task = re.sub(r"\s+", " ", task).strip(" \n\r\t:.-\"'")
    return task


def shorten_reasoning(reasoning: str) -> str:
    reasoning = reasoning.strip()
    if "." in reasoning:
        reasoning = reasoning.split(".")[0].strip() + "."
    return reasoning


def get_speed_ratio(speed: float) -> float:
    denom = MAX_SPEED - MIN_SPEED
    if denom <= 0:
        return 0.5
    return max(0.0, min(1.0, (float(speed) - MIN_SPEED) / denom))


def compute_comfort_zone_relation(current_speed: float, comfort_min: float, comfort_max: float) -> str:
    current_speed = float(current_speed)
    if current_speed < comfort_min:
        return "below_zone"
    if current_speed > comfort_max:
        return "above_zone"
    return "within_zone"


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------
# Fast policy for clear real-time cases
# ---------------------------------------------------------------------

def is_true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def is_false(value: Any) -> bool:
    return value is False or str(value).lower() == "false"


def fast_speed_policy(
    task_context: dict[str, Any],
) -> Optional[dict[str, str]]:
    """
    Handles clear cases without calling CoT.
    Returns None when the case is ambiguous and should be sent to CoT.
    """
    correct = task_context.get("last_trial_correct", None)
    completed = task_context.get("completed_before_next_pick", None)
    relation = task_context.get("comfort_zone_relation", "within_zone")

    success_rate = safe_float(task_context.get("recent_success_rate"), None)
    late_rate = safe_float(task_context.get("recent_late_rate"), None)
    error_rate = safe_float(task_context.get("recent_error_rate"), None)

    completion_time = safe_float(task_context.get("completion_time"), None)
    time_limit = safe_float(task_context.get("time_limit"), None)
    close_to_limit = False
    if completion_time is not None and time_limit is not None and time_limit > 0:
        close_to_limit = completion_time >= 0.90 * time_limit

    if is_true(correct) and is_true(completed) and relation == "within_zone" and not close_to_limit:
        return {
            "reasoning": "Successful on-time performance within the comfort zone supports maintaining speed.",
            "source": "fast_policy",
            "comfort_zone_relation": relation,
            "adaptation_level": "none",
            "speed_decision": "maintain",
        }

    if (
        is_true(correct)
        and is_true(completed)
        and relation == "below_zone"
        and success_rate is not None
        and success_rate >= 0.90
        and (late_rate is None or late_rate <= 0.10)
    ):
        return {
            "reasoning": "Stable on-time success below the comfort zone allows a small speed increase.",
            "source": "fast_policy",
            "comfort_zone_relation": relation,
            "adaptation_level": "small",
            "speed_decision": "increase",
        }

    if relation == "above_zone" and (is_false(correct) or is_false(completed)):
        return {
            "reasoning": "Failure or lateness above the comfort zone is strong evidence of pace pressure.",
            "source": "fast_policy",
            "comfort_zone_relation": relation,
            "adaptation_level": "medium",
            "speed_decision": "decrease",
        }

    if relation == "within_zone" and is_true(correct) and is_false(completed):
        return {
            "reasoning": "Correct but late performance within the comfort zone suggests a conservative decrease.",
            "source": "fast_policy",
            "comfort_zone_relation": relation,
            "adaptation_level": "small",
            "speed_decision": "decrease",
        }

    if error_rate is not None and error_rate >= 0.40:
        return {
            "reasoning": "Recent error rate is high, so a conservative speed decrease is applied.",
            "source": "fast_policy",
            "comfort_zone_relation": relation,
            "adaptation_level": "small" if relation == "within_zone" else "medium",
            "speed_decision": "decrease",
        }

    return None


# ---------------------------------------------------------------------
# CoT speed reasoner
# ---------------------------------------------------------------------

def get_speed_reasoner_prompt(task_context: dict[str, Any], current_speed: Optional[float]) -> str:
    if current_speed is None:
        speed_text = "unknown"
    else:
        try:
            speed_text = f"{float(current_speed):.3f} m/s"
        except (TypeError, ValueError):
            speed_text = "unknown"

    return f"""You are a lightweight reasoning-based speed adaptation planner for a real-time human-robot block sorting task.

An external vision program has already judged the participant's task result.
Use task_context as the source of truth.

Your role is to decide only the robot execution speed adjustment for the next pick-and-place cycle.
Do not change the robot trajectory.
Do not infer task success from the image.

Use the comfort zone as a personalization buffer:
- Within the comfort zone, apply conservative adjustments.
- Outside the comfort zone, failures or delays are stronger evidence of pace mismatch.
- A fast failure within the comfort zone may reflect cognitive confusion rather than speed pressure.
- A late or failed trial above the comfort zone should usually decrease speed.
- A successful timely trial below the comfort zone may allow a small increase.

Available labels:
- AdaptationLevel: none, small, medium
- SpeedDecision: increase, maintain, decrease

Current speed: {speed_text}
Task context: {task_context}

Return exactly four lines:
Reasoning: <one short sentence>
ComfortZoneRelation: <below_zone or within_zone or above_zone>
AdaptationLevel: <none or small or medium>
SpeedDecision: <increase or maintain or decrease>
"""


def parse_speed_reasoner_output(generated_text: str) -> Dict[str, str]:
    reasoning_match = re.search(r"Reasoning:\s*(.*)", generated_text)
    zone_match = re.search(r"ComfortZoneRelation:\s*(.*)", generated_text)
    adaptation_match = re.search(r"AdaptationLevel:\s*(.*)", generated_text)
    speed_match = re.search(r"SpeedDecision:\s*(.*)", generated_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    comfort_zone_relation = zone_match.group(1).strip().lower() if zone_match else ""
    adaptation_level = adaptation_match.group(1).strip().lower() if adaptation_match else ""
    speed_decision = speed_match.group(1).strip().lower() if speed_match else ""

    reasoning = shorten_reasoning(reasoning) if reasoning else ""

    if comfort_zone_relation not in {"below_zone", "within_zone", "above_zone"}:
        comfort_zone_relation = "within_zone"
    if adaptation_level not in {"none", "small", "medium"}:
        adaptation_level = "small"
    if speed_decision not in {"increase", "maintain", "decrease"}:
        speed_decision = "maintain"

    return {
        "reasoning": reasoning,
        "source": "cot_reasoner",
        "comfort_zone_relation": comfort_zone_relation,
        "adaptation_level": adaptation_level,
        "speed_decision": speed_decision,
    }


# ---------------------------------------------------------------------
# Numerical speed controller
# ---------------------------------------------------------------------

def adjust_adaptation_by_comfort_zone(adaptation_level: str, comfort_zone_relation: str) -> str:
    if comfort_zone_relation == "within_zone" and adaptation_level == "medium":
        return "small"
    return adaptation_level


def compute_speed_update_from_decision(
    speed_decision: str,
    adaptation_level: str,
    comfort_zone_relation: str,
    base_speed: Optional[float],
) -> tuple[float, Optional[float], Optional[float], float, str]:
    if base_speed is None:
        return 1.0, base_speed, None, 0.0, adaptation_level

    try:
        base_speed = float(base_speed)
    except (TypeError, ValueError):
        return 1.0, None, None, 0.0, adaptation_level

    speed_ratio = get_speed_ratio(base_speed)
    effective_adaptation_level = adjust_adaptation_by_comfort_zone(
        adaptation_level=adaptation_level,
        comfort_zone_relation=comfort_zone_relation,
    )

    if effective_adaptation_level == "none" or speed_decision == "maintain":
        delta_speed = 0.0
    elif effective_adaptation_level == "small":
        delta_speed = SMALL_SPEED_STEP
    elif effective_adaptation_level == "medium":
        delta_speed = MEDIUM_SPEED_STEP
    else:
        delta_speed = SMALL_SPEED_STEP

    if speed_decision == "decrease":
        delta_speed = -abs(delta_speed)
    elif speed_decision == "increase":
        delta_speed = abs(delta_speed)
    else:
        delta_speed = 0.0

    target_speed = round(clamp_speed(base_speed + delta_speed), 4)
    speed_scale = 1.0

    return speed_scale, target_speed, speed_ratio, round(delta_speed, 4), effective_adaptation_level


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------

class TwoLoopCoTVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        reasoning_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        use_reasoner_gpu: bool = True,
    ) -> None:
        self.openvla_path = openvla_path
        self.reasoning_model_name = reasoning_model_name

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if use_reasoner_gpu and torch.cuda.is_available():
            self.reasoning_device = torch.device("cuda:0")
        else:
            self.reasoning_device = torch.device("cpu")

        self.processor = AutoProcessor.from_pretrained(
            self.openvla_path,
            trust_remote_code=True,
        )

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.vla.eval()

        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(self.reasoning_model_name)

        reasoner_dtype = torch.float16 if self.reasoning_device.type == "cuda" else torch.float32
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(
            self.reasoning_model_name,
            torch_dtype=reasoner_dtype,
            low_cpu_mem_usage=True,
        ).to(self.reasoning_device)
        self.reasoning_model.eval()

        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    self.vla.norm_stats = json.load(f)

        self.user_speed_profiles = load_user_speed_profiles()

        logging.warning(f"[OpenVLA device] {self.device}")
        logging.warning(f"[Reasoner device] {self.reasoning_device}")
        logging.warning(f"[Reasoner model] {self.reasoning_model_name}")
        logging.warning(
            f"[Global comfort zone] {GLOBAL_COMFORT_MIN:.2f}~{GLOBAL_COMFORT_MAX:.2f} "
            f"center={GLOBAL_COMFORT_CENTER:.2f} m/s"
        )
        logging.warning(f"[Loaded user speed profiles] {self.user_speed_profiles}")

        if hasattr(self.vla, "norm_stats"):
            logging.warning(f"[Norm stats keys] {list(self.vla.norm_stats.keys())}")

    def _decode_payload(self, payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        if "encoded" in payload:
            assert len(payload.keys()) == 1, "Only uses encoded payload!"
            return json.loads(payload["encoded"]), True
        return payload, False

    def _json_response(self, result: Dict[str, Any], double_encode: bool = False) -> JSONResponse:
        if double_encode:
            return JSONResponse(json_numpy.dumps(result))
        return JSONResponse(result)

    def reason_about_speed(self, task_context: dict[str, Any], current_speed: Optional[float]) -> Dict[str, str]:
        prompt = get_speed_reasoner_prompt(task_context=task_context, current_speed=current_speed)

        tokenizer_start = time.time()
        inputs = self.reasoning_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.reasoning_device) for k, v in inputs.items()}
        tokenizer_end = time.time()
        print(f"[TIME] Reasoner Tokenize: {tokenizer_end - tokenizer_start:.3f}s")

        generate_start = time.time()
        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=56,
                do_sample=False,
                pad_token_id=self.reasoning_tokenizer.eos_token_id,
            )
        generate_end = time.time()
        print(f"[TIME] Reasoner Generate: {generate_end - generate_start:.3f}s")

        decode_start = time.time()
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.reasoning_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not generated_text.startswith("Reasoning:"):
            generated_text = "Reasoning: " + generated_text
        decode_end = time.time()
        print(f"[TIME] Reasoner Decode: {decode_end - decode_start:.3f}s")

        print("[RAW_SPEED_REASONER_OUTPUT]")
        print(generated_text)

        parse_start = time.time()
        parsed = parse_speed_reasoner_output(generated_text)
        parse_end = time.time()
        print(f"[TIME] Reasoner Parse: {parse_end - parse_start:.3f}s")

        return parsed

    # --------------------------------------------------
    # Loop 1 endpoint: robot / VLA loop
    # --------------------------------------------------
    def act(self, payload: Dict[str, Any]) -> JSONResponse:
        """
        Fast robot loop endpoint.
        This endpoint does not run CoT. It only predicts OpenVLA action and returns
        the latest execution_speed stored in the shared profile state.
        """
        try:
            total_start = time.time()
            payload, double_encode = self._decode_payload(payload)

            image = payload.get("image", None)
            if image is None:
                fallback_path = Path("test.jpg")
                if not fallback_path.exists():
                    raise FileNotFoundError("No image provided and fallback test.jpg not found.")
                image = Image.open(fallback_path).convert("RGB").resize((256, 256))
                image = np.array(image)

            instruction = payload["instruction"]
            user_id = payload.get("user_id", None)
            unnorm_key = payload.get("unnorm_key", "bridge_orig")

            base_speed, has_preferred_speed, base_speed_source = get_base_speed(
                user_id,
                self.user_speed_profiles,
            )
            comfort_min, comfort_max, comfort_center, used_personal_zone, num_samples = (
                compute_personal_comfort_zone(user_id, self.user_speed_profiles)
            )
            comfort_zone_relation = compute_comfort_zone_relation(
                current_speed=base_speed,
                comfort_min=comfort_min,
                comfort_max=comfort_max,
            )

            vla_instruction = sanitize_single_step_task(instruction)
            if not vla_instruction:
                vla_instruction = instruction

            processor_start = time.time()
            prompt = get_openvla_prompt(vla_instruction, self.openvla_path)
            if self.device.type == "cuda":
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device,
                    dtype=torch.bfloat16,
                )
            else:
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device,
                    dtype=torch.float32,
                )
            processor_end = time.time()
            print(f"[TIME] /act OpenVLA Processor: {processor_end - processor_start:.3f}s")

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            vla_start = time.time()
            with torch.no_grad():
                action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            vla_end = time.time()
            print(f"[TIME] /act OpenVLA Inference: {vla_end - vla_start:.3f}s")

            if hasattr(action, "tolist"):
                action = action.tolist()

            result = {
                "endpoint": "/act",
                "loop": "robot_vla_loop",
                "instruction": instruction,
                "vla_instruction": vla_instruction,
                "user_id": user_id,
                "preferred_speed_found": has_preferred_speed,
                "execution_speed": base_speed,
                "current_speed": base_speed,
                "speed_source": base_speed_source,
                "comfort_zone_min": comfort_min,
                "comfort_zone_max": comfort_max,
                "comfort_zone_center": comfort_center,
                "comfort_zone_relation": comfort_zone_relation,
                "personalized_comfort_zone": used_personal_zone,
                "comfort_zone_samples": num_samples,
                "action": action,
            }

            total_end = time.time()
            print(f"[TIME] /act TOTAL: {total_end - total_start:.3f}s")
            print(f"[/act] action trajectory unchanged; execute with speed={base_speed:.3f} m/s")

            return self._json_response(result, double_encode=double_encode)

        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "act failed"}, status_code=500)

    # --------------------------------------------------
    # Loop 2 endpoint: task-performance / speed-adaptation loop
    # --------------------------------------------------
    def update_speed(self, payload: Dict[str, Any]) -> JSONResponse:
        """
        Speed-adaptation endpoint.
        This endpoint receives task_status from an external vision/task monitor,
        updates the next current_speed, and returns reasoning metadata.
        """
        try:
            total_start = time.time()
            payload, double_encode = self._decode_payload(payload)

            user_id = payload.get("user_id", None)
            task_status = payload.get("task_status", {})
            if not isinstance(task_status, dict):
                task_status = {}

            if user_id is not None:
                user_id = str(user_id)
                if user_id not in self.user_speed_profiles or not isinstance(self.user_speed_profiles[user_id], dict):
                    self.user_speed_profiles[user_id] = make_default_profile()

            base_speed, has_preferred_speed, base_speed_source = get_base_speed(
                user_id,
                self.user_speed_profiles,
            )
            comfort_min, comfort_max, comfort_center, used_personal_zone, num_samples = (
                compute_personal_comfort_zone(user_id, self.user_speed_profiles)
            )
            comfort_zone_relation = compute_comfort_zone_relation(
                current_speed=base_speed,
                comfort_min=comfort_min,
                comfort_max=comfort_max,
            )

            task_context = {
                "last_trial_correct": task_status.get("last_trial_correct", None),
                "completed_before_next_pick": task_status.get("completed_before_next_pick", None),
                "completion_time": task_status.get("completion_time", None),
                "time_limit": task_status.get("time_limit", None),
                "recent_success_rate": task_status.get("recent_success_rate", None),
                "recent_late_rate": task_status.get("recent_late_rate", None),
                "recent_error_rate": task_status.get("recent_error_rate", None),
                "current_speed": base_speed,
                "comfort_zone_min": comfort_min,
                "comfort_zone_max": comfort_max,
                "comfort_zone_center": comfort_center,
                "comfort_zone_relation": comfort_zone_relation,
                "personalized_comfort_zone": used_personal_zone,
                "comfort_zone_samples": num_samples,
            }

            fast_start = time.time()
            decision = fast_speed_policy(task_context)
            fast_end = time.time()
            print(f"[TIME] /update_speed Fast Policy: {fast_end - fast_start:.6f}s")

            if decision is None:
                reasoner_start = time.time()
                decision = self.reason_about_speed(
                    task_context=task_context,
                    current_speed=base_speed,
                )
                reasoner_end = time.time()
                print(f"[TIME] /update_speed CoT Reasoner Total: {reasoner_end - reasoner_start:.3f}s")
            else:
                print("[/update_speed] Clear case handled by fast_policy; CoT skipped.")

            speed_decision = decision["speed_decision"]
            adaptation_level = decision["adaptation_level"]
            reasoner_zone_relation = decision.get("comfort_zone_relation", comfort_zone_relation)

            speed_scale, target_speed, speed_ratio, speed_delta, effective_adaptation_level = (
                compute_speed_update_from_decision(
                    speed_decision=speed_decision,
                    adaptation_level=adaptation_level,
                    comfort_zone_relation=reasoner_zone_relation,
                    base_speed=base_speed,
                )
            )

            if user_id is not None and target_speed is not None:
                self.user_speed_profiles[user_id]["current_speed"] = clamp_speed(target_speed)
                save_user_speed_profiles(self.user_speed_profiles)

            profile_saved = False
            saved_speed = None
            if (
                user_id is not None
                and target_speed is not None
                and is_true(task_status.get("last_trial_correct", False))
                and is_true(task_status.get("completed_before_next_pick", False))
            ):
                profile_saved = append_manageable_speed_to_profile(
                    user_id=user_id,
                    new_speed=target_speed,
                    profiles=self.user_speed_profiles,
                )
                if profile_saved:
                    saved_speeds_tmp = self.user_speed_profiles[user_id].get("comfortable_speeds", [])
                    saved_speed = saved_speeds_tmp[-1] if saved_speeds_tmp else None

            saved_speeds = self.user_speed_profiles.get(user_id, {}).get("comfortable_speeds", []) if user_id else []
            current_speed_next = self.user_speed_profiles.get(user_id, {}).get("current_speed", None) if user_id else None

            result = {
                "endpoint": "/update_speed",
                "loop": "task_performance_speed_loop",
                "user_id": user_id,
                "task_status": task_status,
                "task_context": task_context,
                "decision_source": decision.get("source", "unknown"),
                "reasoning": decision.get("reasoning", ""),
                "preferred_speed_found": has_preferred_speed,
                "base_speed": base_speed,
                "base_speed_source": base_speed_source,
                "comfort_zone_min": comfort_min,
                "comfort_zone_max": comfort_max,
                "comfort_zone_center": comfort_center,
                "comfort_zone_relation": comfort_zone_relation,
                "reasoner_comfort_zone_relation": reasoner_zone_relation,
                "personalized_comfort_zone": used_personal_zone,
                "comfort_zone_samples": num_samples,
                "speed_decision": speed_decision,
                "adaptation_level": adaptation_level,
                "effective_adaptation_level": effective_adaptation_level,
                "speed_scale": speed_scale,
                "target_speed": target_speed,
                "execution_speed_next": target_speed,
                "speed_ratio": speed_ratio,
                "speed_delta": speed_delta,
                "profile_saved": profile_saved,
                "saved_speed": saved_speed,
                "saved_speeds": saved_speeds,
                "current_speed_next": current_speed_next,
            }

            total_end = time.time()
            print(f"[TIME] /update_speed TOTAL: {total_end - total_start:.3f}s")
            print(
                f"[/update_speed] user={user_id} | source={result['decision_source']} | "
                f"base={base_speed:.3f} -> target={target_speed} | "
                f"decision={speed_decision}/{effective_adaptation_level} | saved={profile_saved}"
            )

            return self._json_response(result, double_encode=double_encode)

        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "update_speed failed"}, status_code=500)

    def get_speed_state(self, user_id: Optional[str] = None) -> JSONResponse:
        if user_id is None:
            return JSONResponse({"profiles": self.user_speed_profiles})

        user_id = str(user_id)
        base_speed, has_preferred_speed, base_speed_source = get_base_speed(
            user_id,
            self.user_speed_profiles,
        )
        comfort_min, comfort_max, comfort_center, used_personal_zone, num_samples = (
            compute_personal_comfort_zone(user_id, self.user_speed_profiles)
        )
        return JSONResponse({
            "user_id": user_id,
            "current_speed": base_speed,
            "speed_source": base_speed_source,
            "preferred_speed_found": has_preferred_speed,
            "comfort_zone_min": comfort_min,
            "comfort_zone_max": comfort_max,
            "comfort_zone_center": comfort_center,
            "personalized_comfort_zone": used_personal_zone,
            "comfort_zone_samples": num_samples,
            "profile": self.user_speed_profiles.get(user_id, make_default_profile()),
        })

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.act)
        self.app.post("/update_speed")(self.update_speed)
        self.app.get("/speed_state")(self.get_speed_state)
        uvicorn.run(self.app, host=host, port=port)


# ---------------------------------------------------------------------
# Config / entrypoint
# ---------------------------------------------------------------------

@dataclass
class DeployConfig:
    openvla_path: Union[str, Path] = "openvla/openvla-7b"
    reasoning_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    use_reasoner_gpu: bool = True


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = TwoLoopCoTVLAServer(
        openvla_path=cfg.openvla_path,
        reasoning_model_name=cfg.reasoning_model_name,
        use_reasoner_gpu=cfg.use_reasoner_gpu,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()