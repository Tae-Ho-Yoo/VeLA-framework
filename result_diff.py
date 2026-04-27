"""
compare_actions.py

사용법:
  1. OPENVLA_BY_TASK 에 task별 순수 OpenVLA 결과 넣기
  2. CASES_POS / CASES_NEG 에 프로젝트 결과 넣기
  3. python compare_actions.py
"""

import numpy as np

LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# ✅ task별 순수 OpenVLA 결과 (정답)
OPENVLA_BY_TASK = {
    "extend the arm to reach for the cup": np.array([ 0.0025,  0.0067,  0.0060, -0.0193, -0.0108,  0.0418, 0.9961]),
    "reach out to take the cup":           np.array([-0.0067, -0.0027, -0.0117, -0.0097,  0.0314, -0.0113, 0.9961]),
    "extend arm to reach for the cup":     np.array([ 0.0025, -0.0108, -0.0112, -0.0033,  0.0180, -0.0049, 0.9961]),
    "approach the cup":                    np.array([-0.0065, -0.0114,  0.0260, -0.0104, -0.0222, -0.0226, 0.9961]),
}

# ✅ 프로젝트 positive 결과 (name, task, action)
CASES_POS = [
    ("s=0.1", "extend the arm to reach for the cup", [ 0.0025,  0.0067,  0.0060, -0.0193, -0.0108,  0.0418, 0.9961]),
    ("s=0.2", "reach out to take the cup",           [-0.0067, -0.0027, -0.0117, -0.0097,  0.0314, -0.0113, 0.9961]),
    ("s=0.3", "reach out to take the cup",           [-0.0067, -0.0027, -0.0117, -0.0097,  0.0314, -0.0113, 0.9961]),
    ("s=0.4", "reach out to take the cup",           [-0.0067, -0.0027, -0.0117, -0.0097,  0.0314, -0.0113, 0.9961]),
    ("s=0.5", "reach out to take the cup",           [-0.0067, -0.0027, -0.0117, -0.0097,  0.0314, -0.0113, 0.9961]),
]

# ✅ 프로젝트 negative 결과 (name, task, action)
CASES_NEG = [
    ("s=0.1", "approach the cup",                [-0.0110, -0.0194,  0.0442, -0.0176, -0.0378, -0.0385, 0.9961]),
    ("s=0.2", "approach the cup",                [-0.0076, -0.0134,  0.0306, -0.0122, -0.0261, -0.0266, 0.9961]),
    ("s=0.3", "extend arm to reach for the cup", [ 0.0025, -0.0108, -0.0112, -0.0033,  0.0180, -0.0049, 0.9961]),
    ("s=0.4", "extend arm to reach for the cup", [ 0.0023, -0.0098, -0.0102, -0.0030,  0.0164, -0.0045, 0.9961]),
    ("s=0.5", "extend arm to reach for the cup", [ 0.0021, -0.0093, -0.0096, -0.0029,  0.0155, -0.0042, 0.9961]),
]


def mae(a, b):
    return float(np.mean(np.abs(np.array(a[:6]) - np.array(b[:6]))))


def print_table(title, cases):
    names = [c[0] for c in cases]
    tasks  = [c[1] for c in cases]
    projs  = [np.array(c[2]) for c in cases]
    refs   = [OPENVLA_BY_TASK[t] for t in tasks]

    W = 10  # 컬럼 너비
    N = len(cases)

    print()
    print("=" * (14 + (W + 2) * N))
    print(f"  {title}")
    print("=" * (14 + (W + 2) * N))

    # task 행
    print(f"  {'task':12s}", end="")
    for t in tasks:
        short = t[:W]
        print(f"  {short:>{W}}", end="")
    print()

    # speed 행
    print(f"  {'speed':12s}", end="")
    for n in names:
        print(f"  {n:>{W}}", end="")
    print()

    print(f"  {'-' * (12 + (W + 2) * N)}")

    # 축별 정답 / 예측 / 오차
    for i, lbl in enumerate(LABELS):
        if i == 6:
            print(f"  {'-' * (12 + (W + 2) * N)}")

        # 정답
        print(f"  {lbl+' (정답)':12s}", end="")
        for ref in refs:
            print(f"  {ref[i]:>{W}.4f}", end="")
        print()

        # 예측
        print(f"  {'  (예측)':12s}", end="")
        for proj in projs:
            print(f"  {proj[i]:>{W}.4f}", end="")
        print()

        # 오차
        print(f"  {'  (오차)':12s}", end="")
        for ref, proj in zip(refs, projs):
            d = proj[i] - ref[i]
            mark = f"{d:>+{W}.4f}"
            print(f"  {mark}", end="")
        print()

        if i < len(LABELS) - 1:
            print()

    print(f"  {'-' * (12 + (W + 2) * N)}")

    # L1 오차
    print(f"  {'L1 오차':12s}", end="")
    for ref, proj in zip(refs, projs):
        print(f"  {mae(proj, ref):>{W}.4f}", end="")
    print()


# ─── [1] OpenVLA 원본 task별 비교 ───────────────────────────
tasks  = list(OPENVLA_BY_TASK.keys())
shorts = ["extend arm...", "reach out...", "extend arm(2)...", "approach..."]
W = 16
N = len(tasks)

print()
print("=" * (14 + (W + 2) * N))
print("  [1] OpenVLA 원본 — task별 비교")
print("=" * (14 + (W + 2) * N))
print(f"  {'task':12s}", end="")
for s in shorts:
    print(f"  {s:>{W}}", end="")
print()
print(f"  {'-' * (12 + (W + 2) * N)}")

for i, lbl in enumerate(LABELS):
    if i == 6:
        print(f"  {'-' * (12 + (W + 2) * N)}")
    print(f"  {lbl:12s}", end="")
    for t in tasks:
        print(f"  {OPENVLA_BY_TASK[t][i]:>{W}.4f}", end="")
    print()

print(f"  {'-' * (12 + (W + 2) * N)}")
print(f"  {'L1 오차':12s}", end="")
base = OPENVLA_BY_TASK[tasks[0]]
for i, t in enumerate(tasks):
    m = "—" if i == 0 else f"{mae(OPENVLA_BY_TASK[t], base):.4f}"
    print(f"  {m:>{W}}", end="")
print()

# ─── [2] positive 비교 ──────────────────────────────────────
print_table("[2] positive — 정답(OpenVLA) vs 예측(프로젝트)", CASES_POS)

# ─── [3] negative 비교 ──────────────────────────────────────
print_table("[3] negative — 정답(OpenVLA) vs 예측(프로젝트)", CASES_NEG)