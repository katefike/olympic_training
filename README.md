# olympic-training

This repo contains my fitness data. Maybe one day it will get me to the olympics.

## Usage

### HEIC → JSON extraction (`__main__.py`)

Notebook photos live in `heics/` (e.g. `IMG_6428.HEIC`). The script reads each `.heic` file, sends it to an OpenAI **vision** model, and writes one JSON file per image under `automated_data/`, using the same overall shape as the hand-curated files in `training_set/` (a JSON array of workout sessions). Filenames use the photo number: `IMG_6428.HEIC` → `automated_data/6428.json`.

**Prerequisites**

- Python 3
- A virtualenv at the repo root (`.venv`) is recommended.

**Setup**

```bash
cd /path/to/olympic_training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**OpenAI API**

- Create an API key at [platform.openai.com](https://platform.openai.com/). Usage is billed per request (separate from a ChatGPT Plus subscription).
- Export the key before running:

```bash
export OPENAI_API_KEY=sk-...
```

Optional: override the vision model (default is `gpt-4o-mini`):

```bash
export OPENAI_VISION_MODEL=gpt-4o
```

**Run**

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
python __main__.py --limit 1
```

**Useful options**

| Flag | Meaning |
|------|---------|
| `--heic-from N` / `--heic-to M` | Only process `IMG_<id>.heic` files whose numeric id is between `N` and `M` inclusive (both flags required together). Applied before `--limit`. |
| `--example-heic-id N` | Load `heics/IMG_<N>.*` and `training_set/<N>.json` as a few-shot reference to help the model match your JSON formatting. |
| `--limit N` | Process only the first `N` images (good for a cheap test run). |
| `--model NAME` | Vision model id (overrides `OPENAI_VISION_MODEL` and the default). |
| `--sleep SECONDS` | Pause between API calls to reduce rate-limit pressure. |

Examples:

```bash
python3 __main__.py --limit 1
python3 __main__.py --heic-from 6408 --heic-to 6716
python3 __main__.py --example-heic-id 6428 --heic-from 6408 --heic-to 6410
python3 __main__.py --model gpt-4o --sleep 0.5
```

If a request fails for one file, the script still writes `automated_data/<id>.json` as an empty array `[]` and prints an error for that file on stderr.

### Data layout

| Path | Role |
|------|------|
| `heics/` | Source `.heic` images |
| `training_set/` | Reference JSON (manually corrected) |
| `automated_data/` | JSON produced by `__main__.py` |
