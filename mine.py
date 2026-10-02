"""
mine.py — Grabbit Mining Co. / Edge Case Miner

For each scenario: expand into 3-5 search queries (W&B Inference), search every camera
(VSS /api/v1/search), fetch Cosmos caption + YOLO counts + playback URL (VSS /videos/*),
score each hit (W&B Inference), and write results.json in the sample_results.json schema.

Every remote call has a 12 s timeout. When one fails, the hit gets path="fallback" and a
keyword-rule severity; otherwise path="live".

Usage:
  .venv/bin/python mine.py            # all three scenarios
  .venv/bin/python mine.py --first    # first scenario only
"""
import argparse
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import requests
import weave
from openai import OpenAI

TIMEOUT_S = 12
TOP_PER_SCENARIO = 12
SCENARIOS = [
    "forklift within 2m of a worker in an aisle",
    "person close to a moving vehicle",
    "vehicle braking hard",
]
WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_MODEL = os.environ.get("WANDB_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONFIG_PATH = "/config/team-45.config"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def load_team_config():
    if os.environ.get("INGRESS_URL") and os.environ.get("USERNAME") and os.environ.get("PASSWORD"):
        return
    configs = [f for f in os.listdir("/config") if f.endswith(".config")] if os.path.isdir("/config") else []
    path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else (os.path.join("/config", configs[0]) if len(configs) == 1 else None)
    if not path:
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_team_config()
BACKEND = os.environ.get("INGRESS_URL", "").rstrip("/")
WANDB_TEAM = os.environ.get("WANDB_TEAM", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")

llm = OpenAI(
    base_url=WANDB_BASE_URL,
    api_key=os.environ.get("WANDB_API_KEY", "missing"),
    project=f"{WANDB_TEAM}/{WANDB_PROJECT}" if WANDB_TEAM and WANDB_PROJECT else None,
    timeout=TIMEOUT_S,
    max_retries=0,
)


def init_weave():
    """weave.init has no timeout of its own; give it 12 s and continue untraced if it fails."""
    result = {}

    def run():
        try:
            weave.init(f"{WANDB_TEAM}/{WANDB_PROJECT}")
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(TIMEOUT_S)
    return result.get("ok", False), result.get("error", "timeout")


def extract_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    return json.loads(m.group(0)) if m else json.loads(text)


# ---------- W&B Inference ops ----------

@weave.op()
def expand_scenario(scenario: str) -> list:
    resp = llm.chat.completions.create(
        model=WANDB_MODEL,
        messages=[
            {"role": "system", "content": (
                "You write search queries for a video-search engine whose index holds one-paragraph "
                "descriptions of short surveillance clips (warehouses, highways, city streets). "
                "Return ONLY a JSON array of 3 to 5 short, distinct, visual search queries. No prose."
            )},
            {"role": "user", "content": scenario},
        ],
        temperature=0.3,
    )
    queries = extract_json(resp.choices[0].message.content)
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not 3 <= len(queries) <= 5:
        raise ValueError(f"expected 3-5 queries, got {len(queries)}")
    return queries


@weave.op()
def score_hit(scenario: str, caption: str, detections: dict) -> dict:
    resp = llm.chat.completions.create(
        model=WANDB_MODEL,
        messages=[
            {"role": "system", "content": (
                "You judge whether a video clip matches a safety scenario. Use only the clip caption "
                "and YOLO object counts. Reply with strict JSON and nothing else: "
                '{"is_match": bool, "severity": 1-5 integer, "tags": [short snake_case strings], '
                '"rationale": "one sentence"}. Severity: 5 = imminent contact, 1 = no risk.'
            )},
            {"role": "user", "content": json.dumps(
                {"scenario": scenario, "caption": caption, "yolo_counts": detections})},
        ],
        temperature=0,
    )
    out = extract_json(resp.choices[0].message.content)
    sev = int(out["severity"])
    if not isinstance(out.get("is_match"), bool) or not 1 <= sev <= 5:
        raise ValueError(f"invalid judgment: {out}")
    return {
        "is_match": out["is_match"],
        "severity": sev,
        "tags": [str(t) for t in out.get("tags", [])],
        "rationale": str(out.get("rationale", "")).strip(),
    }


# ---------- keyword fallbacks ----------

HIGH_RISK = {
    "running": r"\brunn?ing\b|\bruns\b",
    "under_2m": r"under 2 ?m|within 2 ?m|less than 2 ?m|touching",
    "turning_toward": r"turn\w* (toward|towards)",
}
LOW_RISK = {"stationary": r"stationary|parked|not moving"}


def fallback_queries(scenario):
    return [scenario, f"{scenario} near miss", f"{scenario} warehouse or road"]


def keyword_judgment(caption):
    text = (caption or "").lower()
    high = [tag for tag, rx in HIGH_RISK.items() if re.search(rx, text)]
    low = [tag for tag, rx in LOW_RISK.items() if re.search(rx, text)]
    if high:
        sev = 5 if len(high) >= 2 else 4
    elif low:
        sev = 2 if "person" in text or "worker" in text else 1
    else:
        sev = 3
    tags = high or low
    reason = f"Keyword rule matched: {', '.join(tags)}." if tags else "Keyword rule: no risk keywords found."
    return {"is_match": sev >= 4, "severity": sev, "tags": tags, "rationale": reason}


# ---------- VSS retrieval API ----------

def login():
    r = requests.post(f"{BACKEND}/api/v1/auth/login", timeout=TIMEOUT_S,
                      json={"username": os.environ["USERNAME"], "password": os.environ["PASSWORD"]})
    r.raise_for_status()
    return r.json()["access_token"]


def search(token, query):
    r = requests.post(f"{BACKEND}/api/v1/search", timeout=TIMEOUT_S,
                      headers={"Authorization": f"Bearer {token}"},
                      json={"query": query, "top_k": 20, "llm_top_n": 1,
                            "min_similarity": 0.2, "include_public": True})
    r.raise_for_status()
    return r.json().get("results") or []


def get_metadata(token, source):
    r = requests.get(f"{BACKEND}/api/v1/videos/metadata", timeout=TIMEOUT_S,
                     headers={"Authorization": f"Bearer {token}"}, params={"source": source})
    r.raise_for_status()
    return r.json()


def get_detections(token, source):
    r = requests.get(f"{BACKEND}/api/v1/videos/detections", timeout=TIMEOUT_S,
                     headers={"Authorization": f"Bearer {token}"}, params={"source": source})
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json().get("object_counts") or {}


def get_playback_url(token, source):
    r = requests.get(f"{BACKEND}/api/v1/videos/playback-url", timeout=TIMEOUT_S,
                     params={"source": source, "token": token, "expires_in": 86400})
    r.raise_for_status()
    return r.json().get("url")


def parse_counts(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def segment_id(source):
    return os.path.splitext(os.path.basename(source))[0]


# ---------- pipeline ----------

def mine_scenario(token, scenario):
    try:
        queries = expand_scenario(scenario)
        expand_path = "live"
    except Exception as e:
        print(f"  expand failed ({type(e).__name__}: {e}); using fallback queries")
        queries = fallback_queries(scenario)
        expand_path = "fallback"
    print(f"  expand path={expand_path}: {queries}")

    best = {}
    for q in queries:
        try:
            rows = search(token, q) if token else []
            print(f"  search path=live  {len(rows):>2} rows  {q!r}")
        except Exception as e:
            print(f"  search path=fallback ({type(e).__name__}) {q!r}")
            rows = []
        for row in rows:
            src = row.get("source")
            if src and (src not in best or row.get("similarity_score", 0) > best[src].get("similarity_score", 0)):
                best[src] = row
    top = sorted(best.values(), key=lambda r: r.get("similarity_score", 0), reverse=True)[:TOP_PER_SCENARIO]

    hits = []
    for row in top:
        src = row["source"]
        path = "live"
        caption = row.get("reasoning_content") or ""
        detections = parse_counts(row.get("object_counts"))
        video_url = None
        try:
            meta = get_metadata(token, src)
            caption = meta.get("reasoning_content") or caption
        except Exception:
            path = "fallback"
        try:
            detections = get_detections(token, src) or detections
        except Exception:
            path = "fallback"
        try:
            video_url = get_playback_url(token, src)
        except Exception:
            path = "fallback"

        t0 = time.perf_counter()
        try:
            judgment = score_hit(scenario, caption, detections)
        except Exception:
            judgment = keyword_judgment(caption)
            path = "fallback"
        latency_ms = int((time.perf_counter() - t0) * 1000)

        hit = {
            "id": segment_id(src),
            "camera_id": row.get("camera_id"),
            "location": row.get("location"),
            "start_s": row.get("segment_start_sec"),
            "end_s": row.get("segment_end_sec"),
            "caption": caption,
            "detections": detections,
            "llm": judgment,
            "review": "pending",
            "latency_ms": latency_ms,
            "path": path,
        }
        if video_url:
            hit["video_url"] = video_url
        hits.append(hit)
        print(f"  hit {len(hits):>2} path={path:<8} sev={judgment['severity']} match={judgment['is_match']} "
              f"{latency_ms:>5} ms  {hit['camera_id']}  {hit['id'][:60]}")

    return {
        "query": scenario,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "path": expand_path,
        "hits": hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", action="store_true", help="run only the first scenario")
    args = ap.parse_args()

    traced, err = init_weave()
    project_url = f"https://wandb.ai/{WANDB_TEAM}/{WANDB_PROJECT}/weave"
    print(f"Weave: {'tracing to' if traced else f'NOT tracing ({err}); project would be'} {project_url}")
    print(f"Model: {WANDB_MODEL}")

    try:
        token = login()
        print("VSS login path=live")
    except Exception as e:
        token = None
        print(f"VSS login path=fallback ({type(e).__name__})")

    runs = []
    for scenario in (SCENARIOS[:1] if args.first else SCENARIOS):
        print(f"\n== {scenario}")
        runs.append(mine_scenario(token, scenario))

    with open(OUT_PATH, "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    n = sum(len(r["hits"]) for r in runs)
    fb = sum(h["path"] == "fallback" for r in runs for h in r["hits"])
    print(f"\nWrote {OUT_PATH}: {len(runs)} runs, {n} hits ({fb} fallback)")
    print(f"Weave project: {project_url}")


if __name__ == "__main__":
    main()
