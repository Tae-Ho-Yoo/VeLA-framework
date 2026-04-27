from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

MODEL_ID = "openvla/openvla-7b"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

print("Loading model...")
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

vla = vla.to(DEVICE)
vla.eval()

# 테스트 이미지 불러오기
image = Image.open("test.jpg").convert("RGB")

# OpenVLA 기본 프롬프트 형식
instruction = "pick up the cup"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# 입력 생성
if torch.cuda.is_available():
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
else:
    inputs = processor(prompt, image)

# 액션 예측
with torch.no_grad():
    action = vla.predict_action(
        **inputs,
        unnorm_key="bridge_orig",
        do_sample=False
    )

print("Predicted action:")
print(action)
print("Action shape:", action.shape if hasattr(action, "shape") else type(action))