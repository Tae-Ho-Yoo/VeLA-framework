"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models
(through the HF AutoClass API) over a REST API.

This version adds a lightweight Qwen-based reasoning module:

instruction + emotion_history + user_id
    -> user preferred speed lookup
    -> Qwen (Reasoning + Task + SpeedPreference + ComfortState)
    -> OpenVLA(Task only)
    -> action
    -> speed modulation
    -> save preferred speed only when ComfortState is comfortable
"""

import os
import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
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
import re

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

DEFAULT_SPEED = 0.40
MIN_SPEED = 0.10
MAX_SPEED = 0.50
PROFILE_PATH = Path("user_speed_profiles.json")


def clamp_speed(speed: float) -> float:
    return max(MIN_SPEED, min(MAX_SPEED, float(speed)))

# 유저의 선호 속도를 저장해놓은 JSON파일에서 저장해놓은 선호 속도들 가져오기
def load_user_speed_profiles() -> dict[str, float]:
    if not PROFILE_PATH.exists():
        return {}

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        cleaned = {}
        for k, v in data.items():
            try:
                cleaned[str(k)] = clamp_speed(float(v))
            except (TypeError, ValueError):
                continue
        return cleaned
    except Exception:
        logging.warning("[UserSpeedProfiles] Failed to load profile file. Starting with empty profiles.")
        return {}

# 유저의 선호 속도 JSON파일에 저장
def save_user_speed_profiles(profiles: dict[str, float]) -> None:
    safe_profiles = {}
    for k, v in profiles.items():
        try:
            safe_profiles[str(k)] = round(clamp_speed(float(v)), 4)
        except (TypeError, ValueError):
            continue

    tmp_path = PROFILE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(safe_profiles, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, PROFILE_PATH)

# 유저의 선호속도가 JSON파일에 있으면 그 속도로 진행 아니면 default 속도로 진행
def get_base_speed(user_id: str | None, profiles: dict[str, float]):
    if user_id and user_id in profiles:
        return profiles[user_id], True
    return DEFAULT_SPEED, False

# JSON파일에 유저들의 선호속도 저장하기위해 dictionary생성후 선호속도 dictionary에 저장
def update_user_speed_profile(
    user_id: str | None,
    new_speed: float | None,
    profiles: dict[str, float],
) -> bool:
    if user_id is None or new_speed is None:
        return False

    try:
        profiles[user_id] = clamp_speed(float(new_speed))
        save_user_speed_profiles(profiles)
        return True
    except (TypeError, ValueError):
        return False

# OpenVLA에 입력될 prompt
def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# Planner LLM에 입력될 prompt (AVOID HARD RULE)
def get_reasoner_prompt(
    instruction: str,
    emotion_history: list[str],
    current_speed: float | None,
) -> str:
    history_text = ", ".join([str(e).lower().strip() for e in emotion_history]) if emotion_history else "none"

    if current_speed is None:
        speed_text = "unknown"
    else:
        try:
            speed_text = f"{float(current_speed):.3f} m/s"
        except (TypeError, ValueError):
            speed_text = "unknown"

    return f"""You are a robot task planner for human-robot interaction.

You are given:
- a user instruction
- the user's recent emotional history
- the robot's current speed

Speed information:
- Current speed: {speed_text}
- Speed range: {MIN_SPEED:.2f} m/s to {MAX_SPEED:.2f} m/s

Your role is to decide the robot's next response so that the interaction feels natural, comfortable, and well received by the user.

You should:
1. Interpret the user's recent emotional trend.
2. Decide the robot's next immediate action.
3. Decide an appropriate direction for speed adjustment (increase, maintain, or reduce).
4. Infer the user's likely current comfort state.

Guidelines:
- Focus on evaluating the current interaction rather than relying on a default response style.
- The task must describe only the very next physical step that can be executed immediately.
- Do not skip intermediate steps or describe future actions.
- For object-related instructions, the next step should usually be directed toward the object before manipulation or handover.
- The task should clearly specify the immediate target when possible.
- The task should describe what the robot moves toward or acts on, not how fast it moves.
- Do not include speed-related wording in the task.
- Express speed preference separately from the task.

- Consider how the recent emotional trend and the current speed together shape the user's likely experience.
- Recent negative emotion should be treated as a signal that the current speed may need adjustment, not as automatic evidence that slower is better.
- Evaluate whether the current speed seems too low, too high, or reasonably well matched to the interaction.
- Choose the speed direction that would best improve the interaction if the current speed seems mismatched.
- A speed near the lower end of the range can still feel too low for a natural interaction.
- A speed near the upper end of the range can still feel too high for a comfortable interaction.
- Maintaining the current speed is most appropriate only when the present interaction already seems well matched and a change would add little benefit.

- The reasoning should clearly explain whether the current speed seems too low, too high, or already well matched.
- The chosen speed direction should align with that reasoning.
- Avoid vague reasoning that does not identify the issue with the current speed.

Comfort guidelines:
- ComfortState should reflect the user's current likely state based on recent emotional signals, not the robot's intended behavior.
- Do not assume comfort simply because the robot is adjusting.
- When recent emotion suggests discomfort, comfort should be interpreted conservatively.
- Use "comfortable" only when the interaction already appears stable and well tolerated.

Now answer for the current input only.

Instruction: {instruction}
Emotion History: {history_text}
Current speed: {speed_text}

Always output exactly four lines:

Reasoning: <one short sentence>
Task: <one immediate robot action>
SpeedPreference: <maintain or increase or reduce>
ComfortState: <comfortable or uncomfortable or neutral>

Do not output any extra text.
"""

def sanitize_single_step_task(task: str) -> str:
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
            break

    for bad in ["Instruction:", "Reasoning:", "Task:", "Output:", "Refined Instruction:", "ComfortState:"]:
        if bad.lower() in task.lower():
            idx = task.lower().find(bad.lower())
            task = task[:idx].strip()

    task = task.strip(" \n\r\t:.-\"'")
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

    r = (float(speed) - MIN_SPEED) / denom
    return max(0.0, min(1.0, r))


def compute_speed_scale(
    speed_preference: str,
    base_speed: float | None,
    max_step_ratio: float = 0.20,
):
    """
    max_step_ratio:
        한 번의 modulation에서 허용할 최대 변화 비율
        예: 0.20이면 최대 ±20%
    """
    if base_speed is None:
        return 1.0, base_speed, None

    try:
        base_speed = float(base_speed)
    except (TypeError, ValueError):
        return 1.0, None, None

    r = get_speed_ratio(base_speed)

    if speed_preference == "increase":
        delta_ratio = max_step_ratio * (1.0 - r)
        scale = 1.0 + delta_ratio
    elif speed_preference == "reduce":
        delta_ratio = max_step_ratio * r
        scale = 1.0 - delta_ratio
    else:
        scale = 1.0

    target_speed = clamp_speed(base_speed * scale)
    return scale, target_speed, r


def apply_speed_modulation(action, speed_preference: str, base_speed: float | None = None):
    if action is None:
        return action, 1.0, base_speed, None

    scale, target_speed, r = compute_speed_scale(speed_preference, base_speed)
    modulated = list(action)

    for i in range(min(6, len(modulated))):
        modulated[i] *= scale

    return modulated, scale, target_speed, r


def parse_reasoning_task(generated_text: str):
    reasoning_match = re.search(r"Reasoning:\s*(.*)", generated_text)
    task_match = re.search(r"(?:Task|Refined Instruction):\s*(.*)", generated_text)
    speed_match = re.search(r"SpeedPreference:\s*(.*)", generated_text)
    comfort_match = re.search(r"ComfortState:\s*(.*)", generated_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    task = task_match.group(1).strip() if task_match else ""
    speed_preference = speed_match.group(1).strip().lower() if speed_match else ""
    comfort_state = comfort_match.group(1).strip().lower() if comfort_match else ""

    task = sanitize_single_step_task(task)
    reasoning = shorten_reasoning(reasoning) if reasoning else ""

    valid_speed = {"maintain", "increase", "reduce"}
    if speed_preference not in valid_speed:
        speed_preference = "maintain"

    valid_comfort = {"comfortable", "uncomfortable", "neutral"}
    if comfort_state not in valid_comfort:
        comfort_state = "neutral"

    return {
        "reasoning": reasoning,
        "task": task,
        "speed_preference": speed_preference,
        "comfort_state": comfort_state,
    }


class OpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        reasoning_model_name: str = "Qwen/Qwen2-7B-Instruct",
    ) -> Path:
        self.openvla_path = openvla_path
        self.reasoning_model_name = reasoning_model_name

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.reasoning_device = torch.device("cpu")

        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
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
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

        self.user_speed_profiles = load_user_speed_profiles()

        logging.warning(f"[OpenVLA device] {self.device}")
        logging.warning(f"[Reasoner device] {self.reasoning_device}")
        logging.warning(f"[Reasoner model] {self.reasoning_model_name}")
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

        inputs = self.reasoning_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.reasoning_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.reasoning_tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.reasoning_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not generated_text.startswith("Reasoning:"):
            generated_text = "Reasoning: " + generated_text

        print("[RAW_REASONER_OUTPUT]")
        print(generated_text)

        return parse_reasoning_task(generated_text)

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := ("encoded" in payload):
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image, instruction = payload["image"], payload["instruction"]
            user_id = payload.get("user_id", None)

            emotion_history = payload.get("emotion_history", [])
            if not isinstance(emotion_history, list):
                emotion_history = []

            unnorm_key = payload.get("unnorm_key", "bridge_orig")

            base_speed, has_preferred_speed = get_base_speed(user_id, self.user_speed_profiles)

            reasoner_result = self.refine_instruction(instruction, emotion_history, base_speed)
            refined_instruction = reasoner_result["task"]
            speed_preference = reasoner_result["speed_preference"]
            comfort_state = reasoner_result["comfort_state"]

            if not reasoner_result["reasoning"] or not reasoner_result["task"]:
                result = {
                    "instruction": instruction,
                    "user_id": user_id,
                    "emotion_history": emotion_history,
                    "preferred_speed_found": has_preferred_speed,
                    "base_speed": base_speed,
                    "reasoning": reasoner_result["reasoning"],
                    "task": refined_instruction,
                    "speed_preference": speed_preference,
                    "comfort_state": comfort_state,
                    "target_speed": base_speed,
                    "profile_saved": False,
                    "saved_speed": None,
                    "speed_ratio": None,
                    "action": None,
                }
                if double_encode:
                    return JSONResponse(json_numpy.dumps(result))
                return JSONResponse(result)

            prompt = get_openvla_prompt(refined_instruction, self.openvla_path)

            if self.device.type == "cuda":
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device, dtype=torch.bfloat16
                )
            else:
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device, dtype=torch.float32
                )

            action = self.vla.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

            if hasattr(action, "tolist"):
                action = action.tolist()

            modulated_action, speed_scale, target_speed, speed_ratio = apply_speed_modulation(
                action,
                speed_preference,
                base_speed,
            )

            profile_saved = False
            saved_speed = None

            if user_id is not None and target_speed is not None and comfort_state == "comfortable":
                profile_saved = update_user_speed_profile(
                    user_id=user_id,
                    new_speed=target_speed,
                    profiles=self.user_speed_profiles,
                )
                if profile_saved:
                    saved_speed = self.user_speed_profiles[user_id]

            print(
                f"[USER] {user_id} | "
                f"[Comfort] {comfort_state} | "
                f"[BaseSpeed] {base_speed:.3f} | "
                f"[Target] {target_speed} | "
                f"[Saved] {profile_saved}"
            )
            print(
                f"[Task] {refined_instruction} | "
                f"[Pref] {speed_preference} | "
                f"[Reason] {reasoner_result['reasoning']}"
            )

            result = {
                "instruction": instruction,
                "user_id": user_id,
                "emotion_history": emotion_history,
                "preferred_speed_found": has_preferred_speed,
                "base_speed": base_speed,
                "reasoning": reasoner_result["reasoning"],
                "task": refined_instruction,
                "speed_preference": speed_preference,
                "comfort_state": comfort_state,
                "speed_scale": speed_scale,
                "target_speed": target_speed,
                "profile_saved": profile_saved,
                "saved_speed": saved_speed,
                "speed_ratio": speed_ratio,
                "action": modulated_action,
            }

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            return JSONResponse(result)

        except Exception:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str, 'user_id': Optional[str], "
                "'emotion_history': list[str]}\n"
                "You can optionally pass unnorm_key: str to specify the dataset statistics used for "
                "de-normalizing the output actions."
            )
            return JSONResponse({"error": "generation failed"}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


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