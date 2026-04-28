"""Extract structured workout data from HEIC notebook photos using a vision model.

Reads ``heics/*.heic``, writes ``automated_data/<id>.json`` (same shape as ``training_set/``).

Requires ``OPENAI_API_KEY``. Uses a vision-capable model (default ``gpt-4o-mini``) because
handwritten/structured logs need layout understanding, not plain OCR.

  source .venv/bin/activate
  export OPENAI_API_KEY=...
  python __main__.py
  python __main__.py --limit 1
  python __main__.py --heic-from 6408 --heic-to 6716
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
- "pain": array of objects, where each object has:
  - "pain_level": number, or string like "0.5", or null
  - "pain_location": string or null
  Use one object when there is only one pain entry; use multiple objects when multiple pain entries are recorded.
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
- If weight is present and no unit is written, assume "weight_unit" is "lbs".
- Use "weight_unit": "kg" only when kg is explicitly written.
- Use "weight_unit": null only when no weight value is recorded for that exercise.
- If a set duration is written (commonly 0..90), include it as "duration_sec" for that set.
- If the page uses "E 20 20 20" style notation, interpret it as reps each set and add "reps_each": 20 (or visible number).
- If the page uses "T 20 20 20" style notation, interpret it as total reps and add "reps_total": 20 (or visible number).
- If E/T notation is present for an exercise, do not omit reps fields even when set durations are also present.
- If the page says "No pain", set "pain" to [{"pain_level": 0, "pain_location": null}].
- If any text/numbers are crossed out, do not include them in the JSON.
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


def filter_heic_paths_by_numeric_id(
    paths: list[Path], low: int, high: int
) -> list[Path]:
    """Keep paths whose ``IMG_<n>`` id is in ``low``..``high`` (inclusive)."""
    selected: list[Path] = []
    for p in paths:
        hid = heic_id_from_path(p)
        if not hid.isdigit():
            continue
        n = int(hid)
        if low <= n <= high:
            selected.append(p)
    return selected


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

    pain = raw.get("pain")
    if not isinstance(pain, list):
        pain = []
    return {
        "heic_id": heic_id,
        "gym": raw.get("gym"),
        "date": raw.get("date"),
        "pain": pain,
        "cardio_type": raw.get("cardio_type"),
        "cardio_duration_minutes": raw.get("cardio_duration_minutes"),
        "notes": raw.get("notes"),
        "exercises": exercises,
    }


def vision_extract(
    client: OpenAI,
    model: str,
    jpeg_bytes: bytes,
    *,
    example_jpeg_bytes: bytes | None = None,
    example_output_json: dict | None = None,
) -> list[dict]:
    def to_data_url(jpeg: bytes) -> str:
        b64 = base64.standard_b64encode(jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    url = to_data_url(jpeg_bytes)

    extra_messages: list[dict] = []
    if example_jpeg_bytes is not None and example_output_json is not None:
        example_url = to_data_url(example_jpeg_bytes)
        example_json_text = json.dumps(example_output_json, ensure_ascii=False)
        extra_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Example image and its expected JSON transcription (reference only)."},
                    {"type": "image_url", "image_url": {"url": example_url}},
                ],
            },
            {"role": "assistant", "content": example_json_text},
        ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *extra_messages,
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
    parser.add_argument(
        "--heic-from",
        type=int,
        default=None,
        metavar="N",
        help="Only process IMG_<N>.heic files with numeric id >= N (use with --heic-to).",
    )
    parser.add_argument(
        "--heic-to",
        type=int,
        default=None,
        metavar="N",
        help="Only process IMG_<N>.heic files with numeric id <= N (use with --heic-from).",
    )
    parser.add_argument(
        "--example-heic-id",
        type=int,
        default=None,
        metavar="N",
        help="Optional: use heics/IMG_<N>.* + training_set/<N>.json as a few-shot example for formatting/style.",
    )
    args = parser.parse_args()

    if (args.heic_from is None) ^ (args.heic_to is None):
        parser.error("--heic-from and --heic-to must be given together")
    if args.heic_from is not None and args.heic_from > args.heic_to:
        parser.error("--heic-from must be <= --heic-to")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Structured extraction needs a vision-capable model.\n"
            "Example: export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_heic_files(HEICS_DIR)
    if args.heic_from is not None:
        files = filter_heic_paths_by_numeric_id(
            files, args.heic_from, args.heic_to
        )
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        if args.heic_from is not None:
            print(
                f"No .heic files with id {args.heic_from}..{args.heic_to} under {HEICS_DIR}",
                file=sys.stderr,
            )
        else:
            print(f"No .heic files under {HEICS_DIR}")
        return

    example_jpeg_bytes: bytes | None = None
    example_output_json: dict | None = None
    if args.example_heic_id is not None:
        example_id_str = str(args.example_heic_id)

        example_heic_path: Path | None = None
        for p in iter_heic_files(HEICS_DIR):
            if heic_id_from_path(p) == example_id_str:
                example_heic_path = p
                break

        example_json_path = ROOT / "training_set" / f"{example_id_str}.json"
        if example_heic_path is None or not example_json_path.exists():
            print(
                f"Warning: example not loaded (heic_path={example_heic_path}, json={example_json_path}).",
                file=sys.stderr,
            )
        else:
            example_jpeg_bytes = heic_to_jpeg_bytes(example_heic_path)
            raw_records = json.loads(example_json_path.read_text(encoding="utf-8"))
            if not isinstance(raw_records, list):
                print(
                    f"Warning: example json had unexpected shape (expected list): {example_json_path}",
                    file=sys.stderr,
                )
            else:
                # System prompt says: do not include "heic_id" in model output,
                # but the training_set files include it. Strip it from the example.
                records_wo_heic_id: list[dict] = []
                for r in raw_records:
                    if isinstance(r, dict):
                        r2 = dict(r)
                        r2.pop("heic_id", None)
                        # training_set is now expected to already use the new "pain" list format,
                        # but strip legacy keys just in case.
                        r2.pop("pain_level", None)
                        r2.pop("pain_location", None)
                        if not isinstance(r2.get("pain"), list):
                            r2["pain"] = []
                        records_wo_heic_id.append(r2)

                example_output_json = {"workouts": records_wo_heic_id}

    client = OpenAI()

    for i, heic_path in enumerate(files):
        hid = heic_id_from_path(heic_path)
        out_path = OUT_DIR / f"{hid}.json"
        jpeg = heic_to_jpeg_bytes(heic_path)

        try:
            raw_list = vision_extract(
                client,
                args.model,
                jpeg,
                example_jpeg_bytes=example_jpeg_bytes,
                example_output_json=example_output_json,
            )
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
