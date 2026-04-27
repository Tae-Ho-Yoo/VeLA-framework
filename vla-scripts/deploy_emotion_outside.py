"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models
(through the HF AutoClass API) over a REST API.

This version adds a lightweight Qwen-based reasoning module:

instruction + emotion_history
    -> emotion summary
    -> Qwen (Reasoning + Task)
    -> OpenVLA(Task only)
    -> action
"""

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

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def summarize_emotion_history(emotion_history) -> str:
    """
    Convert online emotion history into a short natural-language summary.
    Expected labels: positive / neutral / negative
    """
    if not emotion_history:
        return "The user showed no strong emotional reaction during the current robot action."

    cleaned = []
    for e in emotion_history:
        e = str(e).lower().strip()
        if e in {"positive", "neutral", "negative"}:
            cleaned.append(e)

    if not cleaned:
        return "The user showed no strong emotional reaction during the current robot action."

    first = cleaned[0]
    last = cleaned[-1]

    if all(e == "positive" for e in cleaned):
        return "The user remained comfortable during the current robot action."

    if all(e == "neutral" for e in cleaned):
        return "The user showed no strong emotional change during the current robot action."

    if all(e == "negative" for e in cleaned):
        return "The user remained uncomfortable during the current robot action."

    if first in {"positive", "neutral"} and last == "negative":
        return "The user became uncomfortable during the current robot action."

    if first == "negative" and last in {"neutral", "positive"}:
        return "The user appeared more comfortable during the current robot action."
    
    if first in {"neutral"} and last == "positive":
        return "The user appeared more comfortable during the current robot action."

    return "The user's emotional response changed during the current robot action."

def get_reasoner_prompt(instruction: str, emotion_summary: str) -> str:
    return f"""You are a robot task planner.

Given:
- a user instruction
- the user's recent emotional history
- a summary of the user's current emotional state

Your job:
1. Interpret the user's emotional state.
2. Decide how the robot should perform the next immediate action.
3. Output one short reasoning sentence and one immediate action.

Important principles:

- Emotion should primarily influence how the robot performs an action (e.g., speed, smoothness, confidence), rather than drastically changing the task itself.
- When the user appears more comfortable, the current action is often appropriate. In such cases, prefer continuing the current motion naturally, unless a clear transition is needed.
- When the user appears uncomfortable, prefer safer, slower, or more careful motion.
- Do not abruptly jump to a distant future action. Focus only on the immediate next executable step.
- The task should remain consistent with the original instruction, but can be slightly refined in style.

Guidelines:

- Treat emotional signals as soft guidance, not strict rules.
- Prefer continuity over sudden changes when interaction is going well.
- Only change task stage (e.g., from moving to grasping) if it is clearly appropriate at the current moment.
- Avoid unnecessary changes in speed or behavior if the current interaction is already smooth.

Examples:

Instruction: give me the cup
Emotion History: ['neutral', 'neutral', 'positive']
Emotion Summary: The user appeared more comfortable during the current robot action.
Reasoning: The user's comfort suggests the current motion is appropriate.
Task: continue moving toward the cup smoothly

Instruction: give me the cup
Emotion History: ['neutral', 'neutral', 'neutral']
Emotion Summary: The user shows no strong emotional change.
Reasoning: The user shows no strong reaction, so continue the task normally.
Task: move toward the cup

Instruction: give me the cup
Emotion History: ['neutral', 'negative', 'negative']
Emotion Summary: The user appears slightly uncomfortable with the robot's motion.
Reasoning: The user's discomfort suggests a more careful approach is needed.
Task: move toward the cup more slowly and gently

Now answer for the current input.

Instruction: {instruction}
Emotion Summary: {emotion_summary}

Output exactly two lines:
Reasoning: <one short sentence>
Task: <one immediate robot action>
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

    for bad in ["Instruction:", "Reasoning:", "Task:", "Output:", "Refined Instruction:"]:
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


def parse_reasoning_task(generated_text: str):
    reasoning_match = re.search(r"Reasoning:\s*([^\n\r]+)", generated_text)
    task_match = re.search(r"(?:Task|Refined Instruction):\s*([^\n\r]+)", generated_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    task = task_match.group(1).strip() if task_match else ""

    task = sanitize_single_step_task(task)
    reasoning = shorten_reasoning(reasoning)

    parse_success = bool(reasoning_match or task_match)

    return {
        "reasoning": reasoning,
        "task": task,
        "has_reasoning": bool(reasoning_match),
        "has_task": bool(task_match),
        "parse_success": parse_success,
    }


# === Server Interface ===
class OpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        reasoning_model_name: str = "Qwen/Qwen2-7B-Instruct",
    ) -> Path:
        """
        Input:
            {
                "image": np.ndarray,
                "instruction": str,
                "emotion_history": list[str],   # optional
                "unnorm_key": Optional[str]
            }

        Output:
            {
                "instruction": str,
                "emotion_history": list[str],
                "emotion_summary": str,
                "reasoning": str,
                "refined_instruction": str,
                "action": list[float] or np.ndarray
            }
        """
        self.openvla_path = openvla_path
        self.reasoning_model_name = reasoning_model_name

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.reasoning_device = torch.device("cpu")

        # Load OpenVLA using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.vla.eval()

        # Load Qwen reasoner on CPU
        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(self.reasoning_model_name)
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(
            self.reasoning_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.reasoning_device)
        self.reasoning_model.eval()

        # Load dataset statistics if available
        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

        logging.warning(f"[OpenVLA device] {self.device}")
        logging.warning(f"[Reasoner device] {self.reasoning_device}")
        logging.warning(f"[Reasoner model] {self.reasoning_model_name}")

        if hasattr(self.vla, "norm_stats"):
            logging.warning(f"[Norm stats keys] {list(self.vla.norm_stats.keys())}")

    def refine_instruction(self, instruction: str, emotion_summary: str) -> Dict[str, str]:
        prompt = get_reasoner_prompt(instruction, emotion_summary)

        inputs = self.reasoning_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.reasoning_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=32,
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

        parsed = parse_reasoning_task(generated_text)

        return {
            "reasoning_prompt": prompt,
            "reasoning_generated_text": generated_text,
            "reasoning": parsed["reasoning"],
            "task": parsed["task"],
            "has_reasoning": parsed["has_reasoning"],
            "has_task": parsed["has_task"],
            "parse_success": parsed["parse_success"],
        }

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := ("encoded" in payload):
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image, instruction = payload["image"], payload["instruction"]

            emotion_history = payload.get("emotion_history", [])
            if not isinstance(emotion_history, list):
                emotion_history = []

            emotion_summary = summarize_emotion_history(emotion_history)

            unnorm_key = payload.get("unnorm_key", "bridge_orig")

           # Qwen reasoning module
            reasoner_result = self.refine_instruction(instruction, emotion_summary)
            refined_instruction = reasoner_result["task"]

            print("[Instruction]", instruction)
            print("[EmotionHistory]", emotion_history)
            print("[EmotionSummary]", emotion_summary)
            print("[RawReasonerOutput]", reasoner_result["reasoning_generated_text"])
            print("[ParsedReasoning]", reasoner_result["reasoning"])
            print("[ParsedTask]", refined_instruction)
            print("[HasReasoning]", reasoner_result["has_reasoning"])
            print("[HasTask]", reasoner_result["has_task"])
            print("[ParseSuccess]", reasoner_result["parse_success"])

            if not reasoner_result["has_reasoning"] or not reasoner_result["has_task"]:
                result = {
                    "instruction": instruction,
                    "emotion_history": emotion_history,
                    "emotion_summary": emotion_summary,
                    "raw_reasoner_output": reasoner_result["reasoning_generated_text"],
                    "parsed_reasoning": reasoner_result["reasoning"],
                    "parsed_task": refined_instruction,
                    "has_reasoning": reasoner_result["has_reasoning"],
                    "has_task": reasoner_result["has_task"],
                    "parse_success": reasoner_result["parse_success"],
                    "cot_success": False,
                    "failure_reason": "empty_or_missing_task",
                    "refined_instruction": None,
                    "action": None,
                }

                if double_encode:
                    return JSONResponse(json_numpy.dumps(result))
                return JSONResponse(result)

            # Original OpenVLA inference path
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

            result = {
                "instruction": instruction,
                "emotion_history": emotion_history,
                "emotion_summary": emotion_summary,
                "raw_reasoner_output": reasoner_result["reasoning_generated_text"],
                "parsed_reasoning": reasoner_result["reasoning"],
                "parsed_task": refined_instruction,
                "has_reasoning": reasoner_result["has_reasoning"],
                "has_task": reasoner_result["has_task"],
                "parse_success": reasoner_result["parse_success"],
                "cot_success": True,
                "failure_reason": None,
                "refined_instruction": refined_instruction,
                "action": action,
            }

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            return JSONResponse(result)

        except Exception:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str, 'emotion_history': list[str]}\n"
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