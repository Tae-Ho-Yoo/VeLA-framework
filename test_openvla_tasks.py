"""
test_openvla_tasks.py

여러 task로 OpenVLA 추론 후 결과 비교
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import numpy as np

# ✅ 이미지 경로
IMAGE_PATH = "test.jpg"

# ✅ 비교할 task 목록
TASKS = [
    ("pos_a",  "extend the arm to reach for the cup"),   # positive s=0.1
    ("pos_b~r","reach out to take the cup"),              # positive s=0.2~0.5
    ("neg_hi", "extend arm to reach for the cup"),        # negative s=0.3~0.5
    ("neg_lo", "approach the cup"),                       # negative s=0.1~0.2
]

LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

print("모델 로딩 중...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")
vla.eval()
print("모델 로딩 완료!\n")

image = Image.open(IMAGE_PATH).convert("RGB").resize((256, 256))

results = {}

for name, task in TASKS:
    prompt = f"In: What action should the robot take to {task}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    results[name] = {"task": task, "action": action}

# -------------------------------------------------------
# 결과 출력
# -------------------------------------------------------
names = list(results.keys())
base_name = names[0]
base_action = results[base_name]["action"]

for name, res in results.items():
    action = res["action"]
    diff   = np.array(action) - np.array(base_action)

    print("=" * 56)
    print(f"  [{name}]  task: {res['task']}")
    print("=" * 56)
    print(f"  {'':8s}  {'액션':>8}  {'vs '+base_name:>10}  {'오차':>8}")
    print(f"  {'-'*48}")
    for i, lbl in enumerate(LABELS):
        mark = " <--" if abs(diff[i]) > 1e-6 and i < 6 else ""
        print(f"  {lbl:8s}  {action[i]:>8.4f}  {diff[i]:>+10.4f}{mark}")
    print(f"  {'-'*48}")
    l1 = float(np.mean(np.abs(diff[:6])))
    print(f"  {'L1 오차':8s}  {l1:.4f}")
    print()

# -------------------------------------------------------
# 전체 요약
# -------------------------------------------------------
print("━" * 56)
print("  전체 요약")
print("━" * 56)
print(f"  {'name':12s}  {'task':35s}  {'L1':>6}")
print(f"  {'-'*56}")
for name, res in results.items():
    diff = np.array(res["action"]) - np.array(base_action)
    l1   = float(np.mean(np.abs(diff[:6])))
    print(f"  {name:12s}  {res['task']:35s}  {l1:.4f}")
print()
