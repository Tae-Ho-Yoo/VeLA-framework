"""
It reads the randomized sequence from Sequence Bridge API:
    GET http://localhost:9100/sequence

Flow:
    Web Page
        -> POST http://localhost:9100/sequence
    Sequence Bridge Server
        -> stores the latest sequence
    This Mock Controller
        -> GET http://localhost:9100/sequence
        -> sends each trial to Task Monitor:
           POST http://localhost:9000/trial_start

You manually place QR blocks into the expected slot in front of the camera.

Run:
    python mock_experiment_controller.py

Optional:
    python mock_experiment_controller.py --sequence_url http://localhost:9100/sequence

Fallback:
    python mock_experiment_controller.py --sequence_file current_sequence.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


def get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(url: str, payload: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_sequence_from_url(sequence_url: str) -> dict[str, Any]:
    """
    Load sequence from Sequence Bridge API.

    Expected response can be either:

    1. Direct sequence object:
        {
          "user_id": "user_01",
          "session_index": 1,
          "time_limit": 2.5,
          "sequence": ["B03", "B01", "B04", "B02"]
        }

    2. Wrapped response:
        {
          "ok": true,
          "sequence_data": {
            "user_id": "user_01",
            "session_index": 1,
            "time_limit": 2.5,
            "sequence": ["B03", "B01", "B04", "B02"]
          }
        }
    """
    data = get_json(sequence_url)

    if "sequence_data" in data and isinstance(data["sequence_data"], dict):
        data = data["sequence_data"]

    if "sequence" not in data or not isinstance(data["sequence"], list):
        raise ValueError(
            "Sequence API response must contain a list field named 'sequence'. "
            f"Received: {data}"
        )

    if len(data["sequence"]) == 0:
        raise ValueError(
            "Sequence is empty. Generate and send a sequence from the webpage first."
        )

    return data


def load_sequence_from_file(path: Path) -> dict[str, Any]:
    """
    Optional fallback.
    Useful if you want to test without the webpage or Sequence Bridge server.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "sequence" not in data or not isinstance(data["sequence"], list):
        raise ValueError("Sequence JSON must contain a list field named 'sequence'.")

    if len(data["sequence"]) == 0:
        raise ValueError("Sequence is empty.")

    return data


def expected_slot_for_index(index: int, slot_prefix: str = "slot_") -> str:
    """
    Converts sequence order into expected slot.

    Example:
        sequence = ["B03", "B01", "B04", "B02"]

        index 0, B03 -> slot_1
        index 1, B01 -> slot_2
        index 2, B04 -> slot_3
        index 3, B02 -> slot_4
    """
    return f"{slot_prefix}{index + 1}"


def wait_until_trial_done(
    task_monitor_state_url: str,
    poll_interval: float = 0.2,
    max_wait: float = 10.0,
) -> dict[str, Any]:
    """
    Wait until Task Monitor finalizes the current trial.

    It checks:
        GET http://localhost:9000/state

    The trial is considered done when:
        active == False
    or:
        finalized == True
    """
    start = time.time()

    while True:
        state = get_json(task_monitor_state_url)
        trial_state = state.get("trial_state", {})

        active = trial_state.get("active", False)
        finalized = trial_state.get("finalized", False)

        if finalized or not active:
            return state

        if time.time() - start > max_wait:
            return {
                "timeout": True,
                "last_state": state,
            }

        time.sleep(poll_interval)


def print_sequence_mapping(sequence: list[str]) -> None:
    print("[SEQUENCE MAPPING]")
    for i, block_id in enumerate(sequence):
        print(f"  {block_id} -> {expected_slot_for_index(i)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence_url",
        type=str,
        default="http://localhost:9100/sequence",
        help="Sequence Bridge API URL. Default: http://localhost:9100/sequence",
    )

    parser.add_argument(
        "--sequence_file",
        type=str,
        default=None,
        help=(
            "Optional fallback JSON file. "
            "If provided, this file is used instead of sequence_url."
        ),
    )

    parser.add_argument(
        "--task_monitor_url",
        type=str,
        default="http://localhost:9000",
        help="Task Monitor base URL. Default: http://localhost:9000",
    )

    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        help="Override user_id from sequence data.",
    )

    parser.add_argument(
        "--session_index",
        type=int,
        default=None,
        help="Override session_index from sequence data.",
    )

    parser.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Override time_limit from sequence data.",
    )

    parser.add_argument(
        "--wait_between_trials",
        type=float,
        default=1.0,
        help="Seconds to wait between trials.",
    )

    parser.add_argument(
        "--max_wait_per_trial",
        type=float,
        default=10.0,
        help=(
            "Maximum seconds to wait until Task Monitor finalizes a trial. "
            "If exceeded, this controller sends /next_pick_start."
        ),
    )

    parser.add_argument(
        "--no_pause",
        action="store_true",
        help="If set, start trials immediately without waiting for Enter.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. Load sequence
    # ------------------------------------------------------------
    if args.sequence_file:
        sequence_data = load_sequence_from_file(Path(args.sequence_file))
        sequence_source = f"file: {args.sequence_file}"
    else:
        sequence_data = load_sequence_from_url(args.sequence_url)
        sequence_source = f"url: {args.sequence_url}"

    user_id = args.user_id or sequence_data.get("user_id", "user_01")

    session_index = args.session_index
    if session_index is None:
        session_index = int(sequence_data.get("session_index", 1))

    time_limit = args.time_limit
    if time_limit is None:
        time_limit = float(sequence_data.get("time_limit", 2.5))

    sequence = sequence_data["sequence"]

    # ------------------------------------------------------------
    # 2. Prepare Task Monitor endpoints
    # ------------------------------------------------------------
    trial_start_url = f"{args.task_monitor_url}/trial_start"
    next_pick_start_url = f"{args.task_monitor_url}/next_pick_start"
    state_url = f"{args.task_monitor_url}/state"

    # ------------------------------------------------------------
    # 3. Print loaded info
    # ------------------------------------------------------------
    print("=" * 70)
    print("[MOCK CONTROLLER - WEB API VERSION]")
    print(f"Sequence source: {sequence_source}")
    print()
    print("[LOADED SEQUENCE DATA]")
    print(json.dumps(sequence_data, indent=2, ensure_ascii=False))
    print()
    print_sequence_mapping(sequence)

    print("[INFO] This program does NOT move UR5e.")
    print("[INFO] Manually place each QR block into the expected slot.")
    print("[INFO] Task Monitor must already be running at:", args.task_monitor_url)
    print()

    if not args.no_pause:
        input("Press Enter to start mock trials...")

    # ------------------------------------------------------------
    # 4. Run trials
    # ------------------------------------------------------------
    for i, block_id in enumerate(sequence):
        trial_index = i + 1
        expected_slot = expected_slot_for_index(i)

        print("=" * 70)
        print(f"[TRIAL {trial_index}] block_id={block_id} expected_slot={expected_slot}")
        print("Place the QR block into the expected slot after /trial_start.")

        payload = {
            "user_id": user_id,
            "block_id": block_id,
            "expected_slot": expected_slot,
            "time_limit": time_limit,
            "trial_index": trial_index,
            "session_index": session_index,
        }

        try:
            result = post_json(trial_start_url, payload)
        except Exception as e:
            print("[ERROR] Failed to call Task Monitor /trial_start")
            print(e)
            return

        print("[/trial_start RESPONSE]")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if not result.get("ok", False):
            print("[ERROR] Task Monitor rejected /trial_start.")
            return

        state = wait_until_trial_done(
            task_monitor_state_url=state_url,
            max_wait=args.max_wait_per_trial,
        )

        if state.get("timeout"):
            print("[WARNING] Trial did not finalize within max_wait_per_trial.")
            print("Sending /next_pick_start to mark trial late/incomplete.")

            late_payload = {
                "user_id": user_id,
                "trial_index": trial_index,
            }

            try:
                late_result = post_json(next_pick_start_url, late_payload)
                print("[/next_pick_start RESPONSE]")
                print(json.dumps(late_result, indent=2, ensure_ascii=False))
            except Exception as e:
                print("[ERROR] Failed to call /next_pick_start")
                print(e)
                return
        else:
            print("[TRIAL DONE]")
            print(json.dumps(state, indent=2, ensure_ascii=False))

        if i < len(sequence) - 1:
            print(f"Waiting {args.wait_between_trials}s before next trial...")
            time.sleep(args.wait_between_trials)

    print("=" * 70)
    print("[MOCK CONTROLLER] All trials completed.")


if __name__ == "__main__":
    main()