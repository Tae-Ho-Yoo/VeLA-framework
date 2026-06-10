"""
Small bridge server between the webpage and the mock experiment controller.

Why this is needed:
- A browser webpage cannot safely write current_sequence.json directly to your PC.
- The webpage sends the generated random sequence to this server.
- This server saves it as current_sequence.json.
- The mock controller can read the file or fetch it from this server.

Run:
    python sequence_bridge_server.py --host 0.0.0.0 --port 9100

API:
    POST /sequence
    GET  /sequence
    GET  /health
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class SequencePayload(BaseModel):
    user_id: str = "user_01"
    session_index: int = 1
    time_limit: float = 2.5
    sequence: List[str] = Field(..., min_length=1)
    condition: Optional[str] = None
    group: Optional[str] = None
    description: Optional[str] = None


def create_app(output_path: Path) -> FastAPI:
    app = FastAPI(title="Sequence Bridge Server")

    # Allow local Vite/React webpage to call this API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "output_path": str(output_path)}

    @app.post("/sequence")
    def save_sequence(payload: SequencePayload):
        data = payload.model_dump()
        data["created_at_unix"] = time.time()
        data["slot_rule"] = "sequence index + 1, e.g., sequence[0] -> slot_1"

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        mapping = {
            block_id: f"slot_{i + 1}"
            for i, block_id in enumerate(data["sequence"])
        }

        print("[SEQUENCE SAVED]")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("[MAPPING]")
        print(json.dumps(mapping, indent=2, ensure_ascii=False))

        return {
            "ok": True,
            "message": "sequence saved",
            "path": str(output_path),
            "sequence": data["sequence"],
            "mapping": mapping,
        }

    @app.get("/sequence")
    def get_sequence():
        if not output_path.exists():
            return {"ok": False, "error": f"sequence file not found: {output_path}"}

        data = json.loads(output_path.read_text(encoding="utf-8"))
        mapping = {
            block_id: f"slot_{i + 1}"
            for i, block_id in enumerate(data.get("sequence", []))
        }
        return {"ok": True, "data": data, "mapping": mapping}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--output", type=str, default="current_sequence.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    app = create_app(output_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
