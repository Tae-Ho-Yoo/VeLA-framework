"""
External Task Monitor for deploy_realtime.py.

Role:
- Watches the participant workspace with a camera.
- Detects QR codes attached to blocks.
- Checks whether the current block is inside the expected slot.
- Checks whether it was completed within the time limit.
- Sends task_status to deploy_realtime.py:
    POST http://localhost:8000/update_speed

Run:
    python task_monitor_qr_slots.py --camera 0 --config task_monitor_slots_config.json
"""

import argparse
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class TrialStartRequest(BaseModel):
    user_id: str
    block_id: str
    expected_slot: Optional[str] = None
    correct_bin: Optional[str] = None
    time_limit: float = 2.5
    trial_index: Optional[int] = None
    session_index: Optional[int] = None


class NextPickRequest(BaseModel):
    user_id: str
    trial_index: Optional[int] = None


@dataclass
class TrialState:
    active: bool = False
    finalized: bool = False
    user_id: str = "user_01"
    block_id: str = ""
    expected_slot: str = ""
    time_limit: float = 2.5
    trial_index: Optional[int] = None
    session_index: Optional[int] = None
    start_time: float = 0.0
    completion_time: Optional[float] = None
    last_detected_slot: Optional[str] = None
    last_detected_qr: Optional[str] = None
    stable_correct_count: int = 0
    stable_any_count: int = 0


@dataclass
class MonitorStats:
    recent_results: deque = field(default_factory=lambda: deque(maxlen=10))

    def add_result(self, correct: bool, late: bool) -> None:
        self.recent_results.append({
            "correct": bool(correct),
            "late": bool(late),
            "error": not bool(correct),
        })

    def rates(self) -> dict[str, float]:
        if not self.recent_results:
            return {
                "recent_success_rate": 1.0,
                "recent_late_rate": 0.0,
                "recent_error_rate": 0.0,
            }

        n = len(self.recent_results)
        return {
            "recent_success_rate": round(sum(1 for r in self.recent_results if r["correct"]) / n, 4),
            "recent_late_rate": round(sum(1 for r in self.recent_results if r["late"]) / n, 4),
            "recent_error_rate": round(sum(1 for r in self.recent_results if r["error"]) / n, 4),
        }


DEFAULT_CONFIG = {
    "camera_width": 1280,
    "camera_height": 720,
    "monitor_port": 9000,
    "speed_server_update_url": "http://localhost:8000/update_speed",
    "stable_frames_required": 5,
    "show_window": True,
    "block_to_expected_slot": {
        "B01": "slot_1",
        "B02": "slot_2",
        "B03": "slot_3",
        "B04": "slot_4"
    },
    "slots": {
        "slot_1": {"name": "slot_1", "polygon": [[100, 150], [250, 150], [250, 320], [100, 320]]},
        "slot_2": {"name": "slot_2", "polygon": [[280, 150], [430, 150], [430, 320], [280, 320]]},
        "slot_3": {"name": "slot_3", "polygon": [[460, 150], [610, 150], [610, 320], [460, 320]]},
        "slot_4": {"name": "slot_4", "polygon": [[640, 150], [790, 150], [790, 320], [640, 320]]}
    }
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CONFIG] Created default config: {path}")
        print("[CONFIG] Edit slot polygons for your real camera view.")
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    if "slots" not in config and "bins" in config:
        config["slots"] = config["bins"]

    if "block_to_expected_slot" not in config and "block_to_correct_bin" in config:
        config["block_to_expected_slot"] = config["block_to_correct_bin"]

    return config


def polygon_contains_point(polygon: list[list[int]], point: tuple[float, float]) -> bool:
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def qr_center(points: np.ndarray) -> tuple[float, float]:
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def find_slot_for_point(point: tuple[float, float], slots: dict[str, Any]) -> Optional[str]:
    for slot_name, slot_info in slots.items():
        if polygon_contains_point(slot_info["polygon"], point):
            return slot_name
    return None


def draw_slots(frame: np.ndarray, slots: dict[str, Any]) -> None:
    for slot_name, slot_info in slots.items():
        polygon = np.array(slot_info["polygon"], dtype=np.int32)
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)
        x, y = polygon[0]
        cv2.putText(frame, slot_name, (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class QRSlotTaskMonitor:
    def __init__(self, camera_index: int, config: dict[str, Any]) -> None:
        self.camera_index = camera_index
        self.config = config
        self.slots = config["slots"]
        self.block_to_expected_slot = config.get("block_to_expected_slot", {})
        self.speed_server_update_url = config["speed_server_update_url"]
        self.stable_frames_required = int(config.get("stable_frames_required", 5))
        self.show_window = bool(config.get("show_window", True))

        self.state = TrialState()
        self.stats = MonitorStats()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.qr_detector = cv2.QRCodeDetector()

    def start_trial(
        self,
        user_id: str,
        block_id: str,
        expected_slot: Optional[str],
        correct_bin: Optional[str],
        time_limit: float,
        trial_index: Optional[int],
        session_index: Optional[int],
    ) -> dict[str, Any]:
        final_expected_slot = expected_slot or correct_bin
        if final_expected_slot is None:
            final_expected_slot = self.block_to_expected_slot.get(block_id)

        if not final_expected_slot:
            raise ValueError(
                f"Expected slot is unknown for block_id={block_id}. "
                "Provide expected_slot in /trial_start or define block_to_expected_slot in config."
            )

        if final_expected_slot not in self.slots:
            raise ValueError(f"expected_slot={final_expected_slot} is not defined in config['slots'].")

        with self.lock:
            self.state = TrialState(
                active=True,
                finalized=False,
                user_id=user_id,
                block_id=block_id,
                expected_slot=final_expected_slot,
                time_limit=float(time_limit),
                trial_index=trial_index,
                session_index=session_index,
                start_time=time.time(),
            )

        print(
            f"[TRIAL START] user={user_id} block={block_id} expected_slot={final_expected_slot} "
            f"time_limit={time_limit}s trial={trial_index} session={session_index}"
        )

        return {
            "ok": True,
            "message": "trial started",
            "user_id": user_id,
            "block_id": block_id,
            "expected_slot": final_expected_slot,
            "time_limit": time_limit,
            "trial_index": trial_index,
            "session_index": session_index,
        }

    def next_pick_started(self, user_id: str, trial_index: Optional[int] = None) -> dict[str, Any]:
        with self.lock:
            should_finalize = self.state.active and not self.state.finalized
            same_user = self.state.user_id == user_id
            same_trial = trial_index is None or self.state.trial_index == trial_index

        if should_finalize and same_user and same_trial:
            result = self.finalize_trial(force_late=True, reason="next_pick_started")
            return {"ok": True, "finalized": True, "result": result}

        return {"ok": True, "finalized": False, "message": "no active matching trial"}

    def build_task_status(
        self,
        correct: bool,
        completed_before_next_pick: bool,
        completion_time: Optional[float],
    ) -> dict[str, Any]:
        rates = self.stats.rates()

        with self.lock:
            state = self.state
            return {
                "last_trial_correct": bool(correct),
                "completed_before_next_pick": bool(completed_before_next_pick),
                "completion_time": round(float(completion_time), 4) if completion_time is not None else None,
                "time_limit": round(float(state.time_limit), 4),
                "recent_success_rate": rates["recent_success_rate"],
                "recent_late_rate": rates["recent_late_rate"],
                "recent_error_rate": rates["recent_error_rate"],
                "block_id": state.block_id,
                "expected_slot": state.expected_slot,
                "correct_bin": state.expected_slot,
                "detected_slot": state.last_detected_slot,
                "detected_bin": state.last_detected_slot,
                "detected_qr": state.last_detected_qr,
                "trial_index": state.trial_index,
                "session_index": state.session_index,
            }

    def post_update_speed(self, user_id: str, task_status: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "task_status": task_status,
        }

        try:
            response = requests.post(self.speed_server_update_url, json=payload, timeout=3.0)
            response.raise_for_status()
            result = response.json()
            print(f"[POST /update_speed OK] target_speed={result.get('target_speed')}")
            return result
        except Exception as e:
            print(f"[POST /update_speed ERROR] {e}")
            return {"error": str(e)}

    def finalize_trial(self, force_late: bool = False, reason: str = "auto") -> dict[str, Any]:
        with self.lock:
            if not self.state.active or self.state.finalized:
                return {"ok": False, "message": "no active trial"}

            now = time.time()
            elapsed = now - self.state.start_time

            detected_correct_slot = self.state.last_detected_slot == self.state.expected_slot
            stable_correct = self.state.stable_correct_count >= self.stable_frames_required
            correct = bool(detected_correct_slot and stable_correct)

            if self.state.completion_time is not None:
                completion_time = self.state.completion_time
            elif correct:
                completion_time = elapsed
            else:
                completion_time = None

            completed_before_next_pick = (
                correct
                and completion_time is not None
                and completion_time <= self.state.time_limit
                and not force_late
            )

            late = not completed_before_next_pick
            user_id = self.state.user_id

            self.state.finalized = True
            self.state.active = False

        self.stats.add_result(correct=correct, late=late)
        task_status = self.build_task_status(correct, completed_before_next_pick, completion_time)
        update_response = self.post_update_speed(user_id, task_status)

        result = {
            "ok": True,
            "reason": reason,
            "user_id": user_id,
            "task_status": task_status,
            "update_speed_response": update_response,
        }

        print(
            f"[TRIAL FINALIZED] reason={reason} correct={correct} "
            f"completed_before_next_pick={completed_before_next_pick} "
            f"completion_time={completion_time} "
            f"expected_slot={task_status.get('expected_slot')} "
            f"detected_slot={task_status.get('detected_slot')}"
        )

        return result

    def process_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            active = self.state.active and not self.state.finalized
            block_id = self.state.block_id
            expected_slot = self.state.expected_slot
            start_time = self.state.start_time
            time_limit = self.state.time_limit

        if not active:
            return

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            self.finalize_trial(force_late=True, reason="time_limit_exceeded")
            return

        ok, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(frame)

        found_current_block = False
        detected_slot = None

        if ok and points is not None and decoded_info is not None:
            for data, pts in zip(decoded_info, points):
                if not data:
                    continue

                center = qr_center(pts)
                slot_name = find_slot_for_point(center, self.slots)

                pts_int = np.array(pts, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(frame, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{data} @ {slot_name}", (int(center[0]) + 8, int(center[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if data == block_id:
                    found_current_block = True
                    detected_slot = slot_name
                    break

        with self.lock:
            if found_current_block:
                self.state.last_detected_qr = block_id
                self.state.last_detected_slot = detected_slot
                self.state.stable_any_count += 1

                if detected_slot == expected_slot:
                    self.state.stable_correct_count += 1
                else:
                    self.state.stable_correct_count = 0
            else:
                self.state.stable_any_count = 0
                self.state.stable_correct_count = 0

            stable_correct = self.state.stable_correct_count >= self.stable_frames_required

            if stable_correct and self.state.completion_time is None:
                self.state.completion_time = time.time() - self.state.start_time

        if stable_correct:
            self.finalize_trial(force_late=False, reason="stable_correct_detection")

    def camera_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("camera_width", 1280)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("camera_height", 720)))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        print(f"[CAMERA] Opened camera index {self.camera_index}")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] Failed to read frame")
                time.sleep(0.05)
                continue

            draw_slots(frame, self.slots)
            self.process_frame(frame)

            with self.lock:
                state = self.state
                if state.active:
                    elapsed = time.time() - state.start_time
                    status_text = (
                        f"ACTIVE user={state.user_id} block={state.block_id} "
                        f"expected={state.expected_slot} "
                        f"elapsed={elapsed:.2f}/{state.time_limit:.2f}s "
                        f"stable={state.stable_correct_count}/{self.stable_frames_required}"
                    )
                else:
                    status_text = "IDLE"

            cv2.putText(frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.show_window:
                cv2.imshow("QR Slot Task Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break

        cap.release()
        cv2.destroyAllWindows()


def create_app(monitor: QRSlotTaskMonitor) -> FastAPI:
    app = FastAPI()

    @app.post("/trial_start")
    def trial_start(req: TrialStartRequest):
        try:
            return monitor.start_trial(
                user_id=req.user_id,
                block_id=req.block_id,
                expected_slot=req.expected_slot,
                correct_bin=req.correct_bin,
                time_limit=req.time_limit,
                trial_index=req.trial_index,
                session_index=req.session_index,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/next_pick_start")
    def next_pick_start(req: NextPickRequest):
        return monitor.next_pick_started(
            user_id=req.user_id,
            trial_index=req.trial_index,
        )

    @app.get("/state")
    def state():
        with monitor.lock:
            return {
                "trial_state": monitor.state.__dict__,
                "recent_rates": monitor.stats.rates(),
            }

    @app.post("/force_finalize")
    def force_finalize():
        return monitor.finalize_trial(force_late=True, reason="force_finalize")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", type=str, default="task_monitor_slots_config.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    monitor = QRSlotTaskMonitor(camera_index=args.camera, config=config)

    camera_thread = threading.Thread(target=monitor.camera_loop, daemon=True)
    camera_thread.start()

    app = create_app(monitor)
    uvicorn.run(app, host=args.host, port=int(config.get("monitor_port", 9000)))


if __name__ == "__main__":
    main()
