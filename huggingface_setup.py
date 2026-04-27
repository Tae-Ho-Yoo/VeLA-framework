# test_openvla_only.py
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")
vla.eval()
print("모델 로딩 완료!")

# 여기만 수정
image = Image.open("test.jpg").convert("RGB").resize((256, 256))
prompt = "In: What action should the robot take to reach out to take the cup?\nOut:"

inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

with torch.no_grad():
    action = vla.predict_action(
        **inputs,
        unnorm_key="bridge_orig",
        do_sample=False,
    )

print("\n=== OpenVLA 순수 액션 (7차원) ===")
labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "Gripper"]
for label, val in zip(labels, action):
    print(f"{label:10s}: {val:.4f}")
print(f"\nGripper 상태: {'열림' if action[6] > 0.5 else '닫힘'}")