"""Extract structured workout data from HEIC notebook photos using a vision model.

Reads ``heics/*.heic``, writes ``automated_data/<id>.json`` (same shape as ``training_set/``).

Requires ``OPENAI_API_KEY``. Uses a vision-capable model (default ``gpt-4o-mini``) because
handwritten/structured logs need layout understanding, not plain OCR.

  source .venv/bin/activate
  export OPENAI_API_KEY=...
  python __main__.py
  python __main__.py --limit 1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from pillow_heif import register_heif_opener
from PIL import Image

register_heif_opener()

ROOT = Path(__file__).resolve().parent
HEICS_DIR = ROOT / "heics"
OUT_DIR = ROOT / "automated_data"

_IMG_NUM = re.compile(r"^IMG_(\d+)$", re.IGNORECASE)

SYSTEM_PROMPT = """You transcribe gym workout log pages (often handwritten, shorthand) into JSON.

The image may show ONE or MORE separate workout sessions (e.g. different dates). Output one JSON object per distinct session you can identify.

Return ONLY valid JSON with this exact top-level shape:
{"workouts":[...]}

Each element of "workouts" must have these keys (use null when not visible or not applicable):
- "gym": string or null
- "date": string in YYMMDD form if you can infer it from the page, else null
- "pain_level": number, or string like "0.5", or null
- "pain_location": string or null
- "cardio_type": string or null
- "cardio_duration_minutes": number or null
- "notes": string or null (session-level notes only)
- "exercises": object whose keys are exercise names in lowercase snake_case (keep author shorthand like gg, bg, ab_crunch, leg_press as keys when that is what the page uses)

Each exercise value is an object with:
- "weight_unit": "lbs", "kg", or null
- "notes": optional string (exercise-specific)
- "reps_total": optional number
- "reps_each": optional number
- "sets": array of objects. Each set object may include "weight" (number), "duration_sec" (number), or be {} if only a checkmark/count is shown. Omit keys you cannot read.

Rules:
- Prefer numbers from the page; never invent weights or times.
- If nothing on the page is legible as workout data, return {"workouts":[]}.
- Do not include "heic_id" in your output."""


def heic_id_from_path(path: Path) -> str:
    stem = path.stem
    m = _IMG_NUM.match(stem)
    if m:
        return m.group(1)
    return stem


def iter_heic_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".heic":
            continue
        out.append(p)
    return out


def heic_to_jpeg_bytes(path: Path, max_side: int = 2048, quality: int = 88) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def parse_workouts_json(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, dict) or "workouts" not in data:
        raise ValueError("Expected JSON object with 'workouts' array")
    workouts = data["workouts"]
    if not isinstance(workouts, list):
        raise ValueError("'workouts' must be an array")
    return workouts


def normalize_record(raw: dict, heic_id: str) -> dict:
    exercises = raw.get("exercises")
    if not isinstance(exercises, dict):
        exercises = {}
    return {
        "heic_id": heic_id,
        "gym": raw.get("gym"),
        "date": raw.get("date"),
        "pain_level": raw.get("pain_level"),
        "pain_location": raw.get("pain_location"),
        "cardio_type": raw.get("cardio_type"),
        "cardio_duration_minutes": raw.get("cardio_duration_minutes"),
        "notes": raw.get("notes"),
        "exercises": exercises,
    }


def vision_extract(
    client: OpenAI,
    model: str,
    jpeg_bytes: bytes,
) -> list[dict]:
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    url = f"data:image/jpeg;base64,{b64}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe every workout session visible on this page into the JSON format described.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty model response")
    return parse_workouts_json(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract workout JSON from HEIC images.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (for testing).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        help="OpenAI vision model (default: gpt-4o-mini or OPENAI_VISION_MODEL).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (rate limiting).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Structured extraction needs a vision-capable model.\n"
            "Example: export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_heic_files(HEICS_DIR)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print(f"No .heic files under {HEICS_DIR}")
        return

    client = OpenAI()

    for i, heic_path in enumerate(files):
        hid = heic_id_from_path(heic_path)
        out_path = OUT_DIR / f"{hid}.json"
        jpeg = heic_to_jpeg_bytes(heic_path)

        try:
            raw_list = vision_extract(client, args.model, jpeg)
        except Exception as e:
            print(f"{heic_path.name}: error ({e!r}), writing empty placeholder.", file=sys.stderr)
            raw_list = []

        records = [normalize_record(r, hid) for r in raw_list if isinstance(r, dict)]

        out_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path.relative_to(ROOT)} ({len(records)} session(s))")

        if args.sleep and i + 1 < len(files):
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
