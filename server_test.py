import requests
import json_numpy
json_numpy.patch()
import numpy as np
from PIL import Image

image = Image.open("test.jpg").convert("RGB").resize((256, 256))
image_np = np.array(image)

response = requests.post(
    "http://127.0.0.1:8000/act",
    json={
        "image": image_np,
        "instruction": "give me the cup",
        "user_id": "user_1",
        "emotion_history": ["neutral", "neutral", "positive"],
        "unnorm_key": "bridge_orig",
    }
)

response_json = response.json()

print(
    f"Instruction: {response_json.get('instruction')}\n"
    f"User ID: {response_json.get('user_id')}\n"
    f"Emotion History: {response_json.get('emotion_history')}\n"
    f"Base Speed: {response_json.get('base_speed')}\n"
    f"Reasoning: {response_json.get('reasoning')}\n"
    f"Task: {response_json.get('task')}\n"
    f"Pace State: {response_json.get('pace_state')}\n"
    f"Speed Ratio: {response_json.get('speed_ratio')}\n"
    f"Speed Scale: {response_json.get('speed_scale')}\n"
    f"Speed Delta: {response_json.get('speed_delta')}\n"
    f"Target Speed: {response_json.get('target_speed')}\n"
    f"Comfort State: {response_json.get('comfort_state')}\n"
    f"Comfort Zone Min: {response_json.get('comfort_zone_min')}\n"
    f"Comfort Zone Max: {response_json.get('comfort_zone_max')}\n"
    f"Comfort Zone Center: {response_json.get('comfort_zone_center')}\n"
    f"Personalized Comfort Zone: {response_json.get('personalized_comfort_zone')}\n"
    f"Comfort Zone Samples: {response_json.get('comfort_zone_samples')}\n"
    f"Profile Saved: {response_json.get('profile_saved')}\n"
    f"Saved Speed: {response_json.get('saved_speed')}\n"
    f"Saved Speeds: {response_json.get('saved_speeds')}\n"
    f"Action: {response_json.get('action')}\n"
    f"Error: {response_json.get('error')}"
)