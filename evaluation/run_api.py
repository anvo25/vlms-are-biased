"""Query an API vision-language model on VLMBias and save its raw answers.

Data comes from the Hugging Face dataset (default) or from a metadata JSON written by
`main.py` / `add_titles.py`. Every row is one image + one prompt; the prompt already
asks for the answer in curly brackets, so it is sent unchanged.

Examples
--------
    # 10 rows of the main split, to check the API key and output format
    python evaluation/run_api.py --provider openai --model gpt-4.1 --limit 10

    # Full main split (2,784 rows)
    python evaluation/run_api.py --provider gemini --model gemini-2.5-pro
    python evaluation/run_api.py --provider anthropic --model claude-sonnet-4-20250514

    # Images you generated yourself
    python evaluation/run_api.py --provider openai --model gpt-4.1 \
        --input-file vlms-are-biased-notitle/animals/animals_notitle_metadata.json

Then score the answers with `python evaluation/score.py results/<file>.jsonl`.

Output: one JSON object per line in results/<model>_<split or input name>.jsonl.
Runs resume: rows already answered are skipped, so re-running the same command is safe.
API keys are read from --api-key-path, or else from OPENAI_API_KEY, ANTHROPIC_API_KEY or
GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

DATASET = "anvo25/vlms-are-biased"
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_rows(args) -> list[dict]:
    """Rows as dicts with the PNG bytes under "image_bytes"."""
    keep = ("ID", "image_path", "topic", "sub_topic", "prompt", "ground_truth", "expected_bias",
            "with_title", "type_of_question", "pixel")
    if args.input_file:
        meta = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
        root = Path(args.image_root)
        return [{**{k: m.get(k) for k in keep}, "image_bytes": (root / m["image_path"]).read_bytes()}
                for m in meta]

    from datasets import Image, load_dataset
    ds = load_dataset(DATASET, split=args.split).cast_column("image", Image(decode=False))
    return [{**{k: r.get(k) for k in keep}, "image_bytes": r["image"]["bytes"]} for r in ds]


# ── Model clients ─────────────────────────────────────────────────────────────

class Client:
    """One call: (PNG bytes, prompt) -> response text. Retries transient errors."""

    def __init__(self, provider: str, model: str, api_key: str, temperature: float | None,
                 max_tokens: int):
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            from google import genai
            self.client = genai.Client(api_key=api_key)

    def _ask(self, png: bytes, prompt: str) -> str:
        b64 = base64.b64encode(png).decode()
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        if self.provider == "openai":
            r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}], **extra)
            return r.choices[0].message.content or ""
        if self.provider == "anthropic":
            r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, messages=[
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}], **extra)
            return "".join(b.text for b in r.content if b.type == "text")
        from google.genai import types
        r = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(**extra))
        return r.text or ""

    def ask(self, png: bytes, prompt: str, retries: int = 3, delay: int = 30) -> str:
        for attempt in range(retries):
            try:
                return self._ask(png, prompt)
            except Exception as e:  # noqa: BLE001 -- any API error is retried, then reported
                if attempt == retries - 1:
                    raise
                print(f"  retry {attempt + 1}/{retries - 1} after error: {e}")
                time.sleep(delay)
        raise RuntimeError("unreachable")


# ── Runner ────────────────────────────────────────────────────────────────────

def load_done(path: Path) -> set[str]:
    """IDs already answered without error."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add(rec["ID"])
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=sorted(KEY_ENV))
    p.add_argument("--model", required=True, help="Model ID, e.g. gpt-4.1, gemini-2.5-pro")
    p.add_argument("--split", default="main",
                   help="HF dataset split: main, identification, withtitle, original, "
                        "remove_background_q1q2, remove_background_q3 (default: main)")
    p.add_argument("--input-file", help="Metadata JSON from main.py/add_titles.py instead of the HF dataset")
    p.add_argument("--image-root", default=".", help="Directory that image_path in --input-file is relative to")
    p.add_argument("--output", help="Output .jsonl (default: results/<model>_<split>.jsonl)")
    p.add_argument("--api-key-path", help=f"File with the API key (default: env var {', '.join(KEY_ENV.values())})")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (default: the API's default; the paper used 1.0)")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens (Anthropic only)")
    p.add_argument("--limit", type=int, help="Only run the first N rows")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    key = (Path(args.api_key_path).read_text().strip() if args.api_key_path
           else os.environ.get(KEY_ENV[args.provider]))
    if not key:
        p.error(f"no API key: pass --api-key-path or set {KEY_ENV[args.provider]}")

    rows = load_rows(args)[: args.limit]
    name = Path(args.input_file).stem if args.input_file else args.split
    out = Path(args.output or f"results/{args.model.replace('/', '_')}_{name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [r for r in rows if r["ID"] not in done]
    print(f"{len(rows)} rows, {len(done)} already answered, {len(todo)} to run -> {out}")

    client = Client(args.provider, args.model, key, args.temperature, args.max_tokens)

    def run(row: dict) -> dict:
        rec = {k: v for k, v in row.items() if k != "image_bytes"}
        rec.update(model=args.model, provider=args.provider, temperature=args.temperature)
        try:
            rec.update(response=client.ask(row["image_bytes"], row["prompt"]), error=None)
        except Exception as e:  # noqa: BLE001
            rec.update(response=None, error=str(e))
        return rec

    errors = 0
    with ThreadPoolExecutor(args.concurrency) as pool, out.open("a", encoding="utf-8") as f:
        for fut in tqdm(as_completed([pool.submit(run, r) for r in todo]), total=len(todo)):
            rec = fut.result()
            errors += rec["error"] is not None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # only this thread writes
            f.flush()
    print(f"Done. {errors} errors (re-run the same command to retry them). Score with:\n"
          f"  python evaluation/score.py {out}")


if __name__ == "__main__":
    main()
