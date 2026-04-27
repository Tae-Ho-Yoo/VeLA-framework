"""
deploy_discrete.py

Provide a lightweight server/client implementation for deploying OpenVLA models
(through the HF AutoClass API) over a REST API.

Pipeline:
instruction + emotion_history + user_id
    -> user personalized comfort-zone lookup
    -> base speed = current_speed (if available) / most recently saved preferred speed / default
    -> Qwen (Reasoning + Task + PaceState + ComfortState)
    -> OpenVLA(Task only)
    -> action (unchanged trajectory)
    -> negative-only comfort-zone execution-speed update
    -> update next current speed
    -> save preferred target speeds
"""

import os
import os.path
import json
import logging
import traceback
import re
import time
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import json_numpy
import torch
import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

json_numpy.patch()

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# ---------------------------------------------------------------------
# Speed / comfort zone configuration
# ---------------------------------------------------------------------

DEFAULT_SPEED = 0.10
MIN_SPEED = 0.10
MAX_SPEED = 0.50

# Global fallback comfort zone
GLOBAL_COMFORT_MIN = 0.25
GLOBAL_COMFORT_MAX = 0.35
GLOBAL_COMFORT_CENTER = (GLOBAL_COMFORT_MIN + GLOBAL_COMFORT_MAX) / 2.0

ADAPT_GAIN = 0.35
MAX_DELTA_SPEED = 0.08

# Personalized comfort-zone settings
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
    zone_min = center - half
    zone_max = center + half
    return clamp_zone(zone_min, zone_max)


def make_default_profile() -> dict[str, Any]:
    return {
        "comfortable_speeds": [],
        "current_speed": None,
    }


def load_user_speed_profiles() -> dict[str, dict[str, Any]]:
    """
    New format:
    {
        "user_3": {
            "comfortable_speeds": [0.29, 0.31, 0.30],
            "current_speed": 0.33
        }
    }

    Backward compatible with old format:
    {
        "user_3": 0.30
    }
    """
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
        logging.warning(
            "[UserSpeedProfiles] Failed to load profile file. Starting with empty profiles."
        )
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
    user_id: str | None,
    profiles: dict[str, dict[str, Any]],
) -> tuple[float, float, float, bool, int]:
    """
    Returns:
        zone_min, zone_max, zone_center, used_personal_zone, num_samples
    """
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
    user_id: str | None,
    profiles: dict[str, dict[str, Any]],
) -> tuple[float, bool, str]:
    """
    Priority:
    1. current_speed
    2. most recent preferred speed
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
    if isinstance(speeds, list) and len(speeds) > 0:
        last_speed = speeds[-1]
        try:
            return clamp_speed(float(last_speed)), True, "preferred_speed"
        except (TypeError, ValueError):
            pass

    return DEFAULT_SPEED, False, "default"


def append_comfort_speed_to_profile(
    user_id: str | None,
    new_speed: float | None,
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


def sample_random_speed_outside_zone(
    zone_min: float,
    zone_max: float,
) -> float:
    """
    Sample a random speed within [MIN_SPEED, MAX_SPEED]
    but outside [zone_min, zone_max].
    Return value rounded to 4 decimal places.
    """
    zone_min, zone_max = clamp_zone(zone_min, zone_max)

    candidates = []

    if MIN_SPEED < zone_min:
        candidates.append((MIN_SPEED, zone_min))
    if zone_max < MAX_SPEED:
        candidates.append((zone_max, MAX_SPEED))

    if not candidates:
        return round(DEFAULT_SPEED, 4)

    lengths = [max(0.0, hi - lo) for lo, hi in candidates]
    total = sum(lengths)

    if total <= 1e-8:
        return round(DEFAULT_SPEED, 4)

    r = np.random.uniform(0.0, total)
    acc = 0.0

    for (lo, hi), seg_len in zip(candidates, lengths):
        if acc + seg_len >= r:
            return round(float(np.random.uniform(lo, hi)), 4)
        acc += seg_len

    lo, hi = candidates[-1]
    return round(float(np.random.uniform(lo, hi)), 4)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return (
            f"{SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{instruction.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def get_speed_ratio(speed: float) -> float:
    denom = MAX_SPEED - MIN_SPEED
    if denom <= 0:
        return 0.5

    r = (float(speed) - MIN_SPEED) / denom
    return max(0.0, min(1.0, r))


def sanitize_single_step_task(task: str) -> str:
    """
    Lightweight cleanup only.
    Keeps the planner's intent, but makes the task safer and cleaner for OpenVLA.
    """
    if not task:
        return ""

    lines = task.splitlines()
    if not lines:
        return ""

    task = lines[0].strip()
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

    for bad in [
        "Instruction:",
        "Reasoning:",
        "Task:",
        "Output:",
        "Refined Instruction:",
        "ComfortState:",
        "PaceState:",
    ]:
        bad_lower = bad.lower()
        task_lower = task.lower()
        if bad_lower in task_lower:
            idx = task_lower.find(bad_lower)
            task = task[:idx].strip()

    speed_words = ["slow", "slowly", "fast", "quickly", "gently", "rapidly"]
    for word in speed_words:
        task = re.sub(rf"\b{word}\b", "", task, flags=re.IGNORECASE)

    task = re.sub(r"\s+", " ", task).strip(" \n\r\t:.-\"'")
    return task


def shorten_reasoning(reasoning: str) -> str:
    reasoning = reasoning.strip()
    if "." in reasoning:
        reasoning = reasoning.split(".")[0].strip() + "."
    return reasoning


# ---------------------------------------------------------------------
# Reasoner prompt / parsing
# ---------------------------------------------------------------------

def get_reasoner_prompt(
    instruction: str,
    emotion_history: list[str],
    current_speed: float | None,
) -> str:
    history_text = ", ".join(
        [str(e).lower().strip() for e in emotion_history]
    ) if emotion_history else "none"

    if current_speed is None:
        speed_text = "unknown"
    else:
        try:
            speed_text = f"{float(current_speed):.3f} m/s"
        except (TypeError, ValueError):
            speed_text = "unknown"

    return f"""You are a robot interaction planner for human-robot handover.

You are given:
- a user instruction
- the user's recent emotion history
- the robot's current speed

Your role is to interpret the current interaction naturally and decide the robot's next immediate physical action.

Focus on the present moment.
Use the recent emotion history as soft context, not as a rigid rule.
Judge whether the current speed feels slower than comfortable, reasonably matched, or faster than comfortable.
Also judge whether the user's likely current experience feels comfortable, neutral, or uncomfortable.

Write like a planner, not like a classifier.
Do not explain policies or rules.
Do not describe future corrections or long-term plans.
Just describe the current interaction briefly and choose the next immediate action.

Task requirements:
- The task must be the next immediate physical action only.
- The task must describe a concrete motion the robot can perform right now.
- Do not restate the full user instruction.
- Do not describe the final goal of the interaction.
- For handover-related instructions, prefer the first physical step toward the object or handover setup, not the completed handover.
- Do not include speed words or pacing phrases.
- Do not include explanation, justification, or user state in the task.
- Keep the task short and action-focused.

Output requirements:
- Reasoning must be one short sentence about the current interaction.
- Task must be one immediate physical action the robot can perform right now.
- Task must not contain multiple steps.
- Task must not include speed adverbs such as slowly, quickly, gently, or rapidly.

Available labels:
- PaceState: too_slow, comfortable, too_fast
- ComfortState: comfortable, neutral, uncomfortable

Instruction: {instruction}
Emotion History: {history_text}
Current speed: {speed_text}

Return exactly four lines in this format:

Reasoning: <one short sentence>
Task: <one immediate robot action>
PaceState: <too_slow or comfortable or too_fast>
ComfortState: <comfortable or neutral or uncomfortable>
"""


def parse_reasoning_task(generated_text: str) -> Dict[str, str]:
    reasoning_match = re.search(r"Reasoning:\s*(.*)", generated_text)
    task_match = re.search(r"(?:Task|Refined Instruction):\s*(.*)", generated_text)
    pace_match = re.search(r"PaceState:\s*(.*)", generated_text)
    comfort_match = re.search(r"ComfortState:\s*(.*)", generated_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    task = task_match.group(1).strip() if task_match else ""
    pace_state = pace_match.group(1).strip().lower() if pace_match else ""
    comfort_state = comfort_match.group(1).strip().lower() if comfort_match else ""

    task = sanitize_single_step_task(task)
    reasoning = shorten_reasoning(reasoning) if reasoning else ""

    valid_pace = {"too_fast", "comfortable", "too_slow"}
    if pace_state not in valid_pace:
        pace_state = "comfortable"

    valid_comfort = {"comfortable", "uncomfortable", "neutral"}
    if comfort_state not in valid_comfort:
        comfort_state = "neutral"

    return {
        "reasoning": reasoning,
        "task": task,
        "pace_state": pace_state,
        "comfort_state": comfort_state,
    }


# ---------------------------------------------------------------------
# Continuous personalized comfort-zone controller
# ---------------------------------------------------------------------

def compute_speed_update_continuous(
    pace_state: str,
    emotion_history: list[str] | None,
    base_speed: float | None,
    comfort_center: float,
):
    """
    Update only the desired execution speed.
    The action / trajectory itself is kept unchanged.

    Returns:
        scale: kept as metadata only (always 1.0 for action output)
        target_speed: updated desired execution speed
        speed_ratio: normalized location of base_speed in [MIN_SPEED, MAX_SPEED]
        delta_speed: additive speed change in m/s
    """
    if base_speed is None:
        return 1.0, base_speed, None, 0.0

    try:
        base_speed = float(base_speed)
    except (TypeError, ValueError):
        return 1.0, None, None, 0.0

    speed_ratio = get_speed_ratio(base_speed)

    latest_emotion = None
    if isinstance(emotion_history, list) and len(emotion_history) > 0:
        latest_emotion = str(emotion_history[-1]).lower().strip()

    target_speed = base_speed
    delta_speed = 0.0

    # negative면 pace_state와 상관없이 무조건 comfort center 쪽으로 이동
    if latest_emotion == "negative":
        desired_speed = comfort_center
        error = desired_speed - base_speed
        raw_delta = ADAPT_GAIN * error
        delta_speed = max(-MAX_DELTA_SPEED, min(MAX_DELTA_SPEED, raw_delta))
        target_speed = round(clamp_speed(base_speed + delta_speed), 4)

    # action 자체는 수정하지 않음
    scale = 1.0

    return scale, target_speed, speed_ratio, delta_speed


def apply_speed_modulation(
    action,
    pace_state: str,
    emotion_history: list[str] | None,
    base_speed: float | None = None,
    comfort_center: float = GLOBAL_COMFORT_CENTER,
):
    if action is None:
        return action, 1.0, base_speed, None, 0.0

    scale, target_speed, speed_ratio, delta_speed = compute_speed_update_continuous(
        pace_state,
        emotion_history,
        base_speed,
        comfort_center,
    )

    # action은 그대로 유지
    # 속도 정보만 별도로 반환
    return action, 1.0, target_speed, speed_ratio, delta_speed


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------

class OpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        reasoning_model_name: str = "Qwen/Qwen2-7B-Instruct",
    ) -> None:
        self.openvla_path = openvla_path
        self.reasoning_model_name = reasoning_model_name

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(
            self.reasoning_model_name,
            torch_dtype=torch.float32,
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
            f"[Global comfort zone] {GLOBAL_COMFORT_MIN:.2f} ~ {GLOBAL_COMFORT_MAX:.2f} "
            f"(center={GLOBAL_COMFORT_CENTER:.2f}) m/s"
        )
        logging.warning(f"[Loaded user speed profiles] {self.user_speed_profiles}")

        if hasattr(self.vla, "norm_stats"):
            logging.warning(f"[Norm stats keys] {list(self.vla.norm_stats.keys())}")

    def refine_instruction(
        self,
        instruction: str,
        emotion_history: list[str],
        current_speed: float | None,
    ) -> Dict[str, str]:
        prompt = get_reasoner_prompt(instruction, emotion_history, current_speed)

        tokenizer_start = time.time()
        inputs = self.reasoning_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.reasoning_device) for k, v in inputs.items()}
        tokenizer_end = time.time()
        print(f"[TIME] Reasoner Tokenize: {tokenizer_end - tokenizer_start:.3f}s")

        generate_start = time.time()
        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=96,
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

        print("[RAW_REASONER_OUTPUT]")
        print(generated_text)

        parse_start = time.time()
        parsed = parse_reasoning_task(generated_text)
        parse_end = time.time()
        print(f"[TIME] Reasoner Parse: {parse_end - parse_start:.3f}s")

        return parsed

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            total_start = time.time()

            if double_encode := ("encoded" in payload):
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            preprocess_start = time.time()

            image = payload.get("image", None)

            if image is None:
                fallback_path = Path("test.jpg")
                if not fallback_path.exists():
                    raise FileNotFoundError("No image provided and fallback test.jpg not found.")
                image = Image.open(fallback_path).convert("RGB").resize((256, 256))
                image = np.array(image)

            instruction = payload["instruction"]
            user_id = payload.get("user_id", None)

            emotion_history = payload.get("emotion_history", [])
            if not isinstance(emotion_history, list):
                emotion_history = []

            unnorm_key = payload.get("unnorm_key", "bridge_orig")

            comfort_min, comfort_max, comfort_center, used_personal_zone, num_samples = (
                compute_personal_comfort_zone(user_id, self.user_speed_profiles)
            )

            base_speed, has_preferred_speed, base_speed_source = get_base_speed(
                user_id,
                self.user_speed_profiles,
            )

            preprocess_end = time.time()
            print(f"[TIME] Preprocess/Profile Lookup: {preprocess_end - preprocess_start:.3f}s")

            print(
                f"[CURRENT COMFORT ZONE] user={user_id} | "
                f"min={comfort_min:.3f} | max={comfort_max:.3f} | center={comfort_center:.3f} | "
                f"personalized={used_personal_zone} | samples={num_samples}"
            )
            print(
                f"[BASE SPEED] user={user_id} | base_speed={base_speed:.3f} | source={base_speed_source}"
            )

            # --------------------------------------------------
            # 1) Reasoner timing
            # --------------------------------------------------
            reasoner_start = time.time()
            reasoner_result = self.refine_instruction(
                instruction,
                emotion_history,
                base_speed,
            )
            reasoner_end = time.time()
            print(f"[TIME] Reasoner Total: {reasoner_end - reasoner_start:.3f}s")

            refined_instruction = reasoner_result["task"]
            pace_state = reasoner_result["pace_state"]
            comfort_state = reasoner_result["comfort_state"]

            if not reasoner_result["reasoning"] or not refined_instruction:
                total_end = time.time()
                print(f"[TIME] TOTAL: {total_end - total_start:.3f}s")

                result = {
                    "instruction": instruction,
                    "user_id": user_id,
                    "emotion_history": emotion_history,
                    "preferred_speed_found": has_preferred_speed,
                    "base_speed": base_speed,
                    "base_speed_source": base_speed_source,
                    "reasoning": reasoner_result["reasoning"],
                    "task": refined_instruction,
                    "pace_state": pace_state,
                    "comfort_state": comfort_state,
                    "comfort_zone_min": comfort_min,
                    "comfort_zone_max": comfort_max,
                    "comfort_zone_center": comfort_center,
                    "personalized_comfort_zone": used_personal_zone,
                    "comfort_zone_samples": num_samples,
                    "speed_scale": 1.0,
                    "target_speed": base_speed,
                    "execution_speed": base_speed,
                    "profile_saved": False,
                    "saved_speed": None,
                    "saved_speeds": self.user_speed_profiles.get(user_id, {}).get("comfortable_speeds", []) if user_id else [],
                    "current_speed_next": self.user_speed_profiles.get(user_id, {}).get("current_speed", None) if user_id else None,
                    "speed_ratio": None,
                    "speed_delta": 0.0,
                    "action": None,
                }
                if double_encode:
                    return JSONResponse(json_numpy.dumps(result))
                return JSONResponse(result)

            # --------------------------------------------------
            # 2) OpenVLA input processing timing
            # --------------------------------------------------
            processor_start = time.time()
            prompt = get_openvla_prompt(refined_instruction, self.openvla_path)

            if self.device.type == "cuda":
                inputs = self.processor(
                    prompt,
                    Image.fromarray(image).convert("RGB"),
                ).to(self.device, dtype=torch.bfloat16)
            else:
                inputs = self.processor(
                    prompt,
                    Image.fromarray(image).convert("RGB"),
                ).to(self.device, dtype=torch.float32)

            processor_end = time.time()
            print(f"[TIME] OpenVLA Processor: {processor_end - processor_start:.3f}s")

            # --------------------------------------------------
            # 3) OpenVLA inference timing
            # --------------------------------------------------
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            vla_start = time.time()

            action = self.vla.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            vla_end = time.time()
            print(f"[TIME] OpenVLA Inference: {vla_end - vla_start:.3f}s")

            post_action_start = time.time()
            if hasattr(action, "tolist"):
                action = action.tolist()
            post_action_end = time.time()
            print(f"[TIME] Action tolist: {post_action_end - post_action_start:.6f}s")

            # --------------------------------------------------
            # 4) Execution-speed update timing (action kept unchanged)
            # --------------------------------------------------
            modulation_start = time.time()

            modulated_action, speed_scale, target_speed, speed_ratio, speed_delta = (
                apply_speed_modulation(
                    action,
                    pace_state,
                    emotion_history,
                    base_speed,
                    comfort_center,
                )
            )

            modulation_end = time.time()
            print(f"[TIME] Execution Speed Update: {modulation_end - modulation_start:.6f}s")

            # --------------------------------------------------
            # 5) Save / profile update timing
            # --------------------------------------------------
            save_start = time.time()

            profile_saved = False
            saved_speed = None

            latest_emotion = None
            if isinstance(emotion_history, list) and len(emotion_history) > 0:
                latest_emotion = str(emotion_history[-1]).lower().strip()

            if user_id is not None:
                if user_id not in self.user_speed_profiles or not isinstance(self.user_speed_profiles[user_id], dict):
                    self.user_speed_profiles[user_id] = make_default_profile()

            # 1) positive -> save preferred speed, then next speed is random OUTSIDE comfort zone
            if user_id is not None and target_speed is not None and latest_emotion == "positive":
                profile_saved = append_comfort_speed_to_profile(
                    user_id=user_id,
                    new_speed=target_speed,
                    profiles=self.user_speed_profiles,
                )
                if profile_saved:
                    saved_speeds = self.user_speed_profiles[user_id]["comfortable_speeds"]
                    saved_speed = saved_speeds[-1] if saved_speeds else None

                self.user_speed_profiles[user_id]["current_speed"] = sample_random_speed_outside_zone(
                    comfort_min,
                    comfort_max,
                )
                save_user_speed_profiles(self.user_speed_profiles)

            # 2) negative -> next speed becomes the modulated target speed
            elif user_id is not None and target_speed is not None and latest_emotion == "negative":
                self.user_speed_profiles[user_id]["current_speed"] = clamp_speed(target_speed)
                save_user_speed_profiles(self.user_speed_profiles)

            # 3) neutral or others -> keep target speed as next speed
            elif user_id is not None and target_speed is not None:
                self.user_speed_profiles[user_id]["current_speed"] = clamp_speed(target_speed)
                save_user_speed_profiles(self.user_speed_profiles)

            # 4) optional save if comfortable and inside zone (even without positive)
            if (
                user_id is not None
                and target_speed is not None
                and latest_emotion != "positive"
                and comfort_state == "comfortable"
                and comfort_min <= target_speed <= comfort_max
            ):
                profile_saved = append_comfort_speed_to_profile(
                    user_id=user_id,
                    new_speed=target_speed,
                    profiles=self.user_speed_profiles,
                )
                if profile_saved:
                    saved_speeds = self.user_speed_profiles[user_id]["comfortable_speeds"]
                    saved_speed = saved_speeds[-1] if saved_speeds else None

            saved_speeds = self.user_speed_profiles.get(user_id, {}).get("comfortable_speeds", []) if user_id else []
            current_speed_next = self.user_speed_profiles.get(user_id, {}).get("current_speed", None) if user_id else None

            save_end = time.time()
            print(f"[TIME] Profile Save/Update: {save_end - save_start:.6f}s")

            print(
                f"[USER] {user_id} | "
                f"[Comfort] {comfort_state} | "
                f"[PaceState] {pace_state} | "
                f"[BaseSpeed] {base_speed:.3f} | "
                f"[Zone] {comfort_min:.3f}~{comfort_max:.3f} (center={comfort_center:.3f}) | "
                f"[Target] {target_speed} | "
                f"[NextCurrent] {current_speed_next} | "
                f"[Saved] {profile_saved}"
            )
            print("[ACTION] trajectory unchanged; apply target_speed separately in the UR5e execution layer")
            print(
                f"[Task] {refined_instruction} | "
                f"[Reason] {reasoner_result['reasoning']}"
            )

            total_end = time.time()
            print(f"[TIME] TOTAL: {total_end - total_start:.3f}s")

            result = {
                "instruction": instruction,
                "user_id": user_id,
                "emotion_history": emotion_history,
                "preferred_speed_found": has_preferred_speed,
                "base_speed": base_speed,
                "base_speed_source": base_speed_source,
                "reasoning": reasoner_result["reasoning"],
                "task": refined_instruction,
                "pace_state": pace_state,
                "comfort_state": comfort_state,
                "comfort_zone_min": comfort_min,
                "comfort_zone_max": comfort_max,
                "comfort_zone_center": comfort_center,
                "personalized_comfort_zone": used_personal_zone,
                "comfort_zone_samples": num_samples,
                "speed_scale": speed_scale,
                "target_speed": target_speed,
                "execution_speed": target_speed,
                "profile_saved": profile_saved,
                "saved_speed": saved_speed,
                "saved_speeds": saved_speeds,
                "current_speed_next": current_speed_next,
                "speed_ratio": speed_ratio,
                "speed_delta": speed_delta,
                "action": modulated_action,
            }

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            return JSONResponse(result)

        except Exception:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with "
                "the expected format:\n"
                "{'image': np.ndarray, 'instruction': str, 'user_id': Optional[str], "
                "'emotion_history': list[str]}\n"
                "You can optionally pass unnorm_key: str to specify the dataset "
                "statistics used for de-normalizing the output actions."
            )
            return JSONResponse({"error": "generation failed"}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


# ---------------------------------------------------------------------
# Config / entrypoint
# ---------------------------------------------------------------------

@dataclass
class DeployConfig:
    openvla_path: Union[str, Path] = "openvla/openvla-7b"
    reasoning_model_name: str = "Qwen/Qwen2-7B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(
        openvla_path=cfg.openvla_path,
        reasoning_model_name=cfg.reasoning_model_name,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()