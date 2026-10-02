"""
mine.py — Grabbit Mining Co. / Edge Case Miner

For each scenario: expand it into search queries (W&B Inference), search every camera
in the VSS index, pull the Cosmos Reason caption + YOLO counts for each hit, score each
hit with an LLM (W&B Inference), and write results.json in the sample_results.json schema.

Every remote call has a 12 s timeout. If a call for a hit fails, the hit gets
path = "fallback" and its severity comes from a keyword rule on the caption.

Usage:
  python mine.py              -> all three scenarios
  python mine.py --limit 1    -> first scenario only
"""
import argparse
import glob
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from urllib.parse import quote

SCENARIOS = [
    "forklift within 2 m of a worker in an aisle",
    "person close to a moving vehicle",
    "vehicle braking hard",
]
# Per scenario: the objects that make it a match, as (caption words, YOLO classes).
# Forklift is not a YOLO (COCO) class, so only Cosmos can supply that evidence.
PERSON = (["person", "worker", "pedestrian", "people", "man ", "woman"], ["person"])
FORKLIFT = (["forklift"], [])
VEHICLE = (["vehicle", "car", "truck", "bus", "van", "forklift", "motorcycle"],
           ["car", "truck", "bus", "motorcycle"])
SCENARIO_OBJECTS = {
    SCENARIOS[0]: [FORKLIFT, PERSON],
    SCENARIOS[1]: [PERSON, VEHICLE],
    SCENARIOS[2]: [VEHICLE],
}
TIMEOUT_S = 12
TOP_HITS = 12
INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

try:
    import weave
    op = weave.op()
except ImportError:
    weave = None

    def op(fn):
        return fn

try:
    import openai
except ImportError:
    openai = None


def load_env():
    """Env vars win; anything missing is read from the single /config/*.config file."""
    configs = sorted(glob.glob("/config/*.config"))
    if len(configs) == 1:
        with open(configs[0]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------- VSS retrieval API ----------

class Backend:
    def __init__(self):
        self.base = os.environ["INGRESS_URL"].rstrip("/")
        self.token = None

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.load(r)

    def login(self):
        resp = self.call("POST", "/api/v1/auth/login",
                         {"username": os.environ["USERNAME"], "password": os.environ["PASSWORD"]})
        self.token = resp["access_token"]

    def search(self, query):
        return self.call("POST", "/api/v1/search", {
            "query": query,
            "top_k": 20,
            "llm_top_n": 1,
            "min_similarity": 0.2,
            "include_public": True,
        })

    def metadata(self, source):
        return self.call("GET", f"/api/v1/videos/metadata?source={quote(source, safe='')}")

    def detections(self, source):
        return self.call("GET", f"/api/v1/videos/detections?source={quote(source, safe='')}")

    def playback_url(self, source):
        q = f"source={quote(source, safe='')}&token={self.token}&expires_in=3600"
        return self.call("GET", f"/api/v1/videos/playback-url?{q}")


# ---------- W&B Inference ----------

def inference_client():
    if openai is None:
        return None
    return openai.OpenAI(
        base_url=INFERENCE_BASE_URL,
        api_key=os.environ["WANDB_API_KEY"],
        project=f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}",
        timeout=TIMEOUT_S,
        max_retries=0,
    )


def chat(client, system, user):
    if client is None:
        raise RuntimeError("openai package not installed")
    resp = client.chat.completions.create(
        model=INFERENCE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    return resp.choices[0].message.content


def parse_json_object(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in LLM output: {text[:120]!r}")
    return json.loads(match.group(0))


@op
def expand_scenario(client, scenario: str) -> list:
    system = ("You turn a description of a dangerous moment into search queries for a video "
              "archive of warehouse, street and highway cameras. Reply with JSON only: "
              '{"queries": ["...", "..."]} containing 3 to 5 short, distinct queries.')
    queries = parse_json_object(chat(client, system, scenario))["queries"]
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not 3 <= len(queries) <= 5:
        raise ValueError(f"expected 3-5 queries, got {len(queries)}")
    return queries


@op
def judge_hit(client, scenario: str, caption: str, detections: dict) -> dict:
    system = ("You review one video segment against a safety scenario. Reply with strict JSON "
              'only, no prose: {"is_match": true|false, "severity": 1-5, "tags": ["..."], '
              '"rationale": "one sentence"}. Severity 5 = imminent contact, 1 = no risk.')
    user = (f"Scenario: {scenario}\nCosmos Reason description: {caption}\n"
            f"YOLO object counts: {json.dumps(detections)}")
    out = parse_json_object(chat(client, system, user))
    if not isinstance(out.get("is_match"), bool):
        raise ValueError("is_match must be a bool")
    if not isinstance(out.get("severity"), int) or not 1 <= out["severity"] <= 5:
        raise ValueError("severity must be an int 1-5")
    if not isinstance(out.get("tags"), list) or not isinstance(out.get("rationale"), str):
        raise ValueError("tags must be a list and rationale a string")
    return {"is_match": out["is_match"], "severity": out["severity"],
            "tags": [str(t) for t in out["tags"]], "rationale": out["rationale"].strip()}


# ---------- keyword fallback ----------

HIGH_RISK = {
    "running": ["running", " runs ", "ran toward"],
    "under_2m": ["under 2 m", "under 2m", "within 2 m", "within 2m", "touching"],
    "turning_toward": ["turning toward", "turns toward", "turning towards", "turns towards"],
}
LOW_RISK = ["stationary", "parked", "not moving"]


def keyword_judgment(caption):
    text = f" {caption.lower()} "
    tags = [tag for tag, words in HIGH_RISK.items() if any(w in text for w in words)]
    if tags:
        severity = 5 if len(tags) >= 2 else 4
    elif any(w in text for w in LOW_RISK):
        severity = 2 if "person" in text else 1
        tags = ["stationary"]
    else:
        severity = 3
    return {"is_match": severity >= 4, "severity": severity, "tags": tags,
            "rationale": f"Keyword rule (LLM unavailable): severity {severity} from caption keywords."}


# ---------- pipeline ----------

def segment_id(source):
    return os.path.basename(source).rsplit(".", 1)[0]


def parse_counts(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def evidence_source(scenario, caption, detections):
    """Which signal covers all of the scenario's key objects. Otherwise the one covering
    more of them; ties go to cosmos, since the LLM severity is judged from the caption."""
    text = f" {caption.lower()} "
    objects = SCENARIO_OBJECTS.get(scenario, [])
    cosmos = sum(any(w in text for w in words) for words, _ in objects)
    yolo = sum(any(detections.get(c) for c in classes) for _, classes in objects)
    if objects and cosmos == yolo == len(objects):
        return "both"
    return "yolo" if yolo > cosmos else "cosmos"


def build_hit(backend, client, scenario, row, rank):
    source = row["source"]
    failures = []
    hit = {
        "id": segment_id(source),
        "search_rank": rank,
        "similarity": round(row["similarity_score"], 4),
        "camera_id": row.get("camera_id"),
        "location": row.get("location"),
        "start_s": row.get("segment_start_sec"),
        "end_s": row.get("segment_end_sec"),
        "caption": row.get("reasoning_content") or "",
        "detections": parse_counts(row.get("object_counts")),
    }

    try:
        meta = backend.metadata(source)
        hit["caption"] = meta.get("reasoning_content") or hit["caption"]
    except Exception as e:
        failures.append(f"metadata: {e}")

    try:
        hit["detections"] = parse_counts(backend.detections(source).get("object_counts")) or hit["detections"]
    except urllib.error.HTTPError as e:
        if e.code != 404:  # 404 = no YOLO sidecar for this segment, not a failure
            failures.append(f"detections: {e}")
    except Exception as e:
        failures.append(f"detections: {e}")

    try:
        url = backend.playback_url(source).get("url")
        if url:
            hit["video_url"] = url
    except Exception as e:
        failures.append(f"playback-url: {e}")

    t0 = time.monotonic()
    try:
        hit["llm"] = judge_hit(client, scenario, hit["caption"], hit["detections"])
    except Exception as e:
        failures.append(f"llm: {e}")
        hit["llm"] = keyword_judgment(hit["caption"])
    hit["latency_ms"] = round((time.monotonic() - t0) * 1000)
    hit["evidence_source"] = evidence_source(scenario, hit["caption"], hit["detections"])

    hit["review"] = "pending"
    hit["path"] = "fallback" if failures else "live"
    if failures:
        hit["fallback_reason"] = "; ".join(failures)[:300]
    return hit


def mine(backend, client, scenario):
    run = {"query": scenario, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}

    try:
        queries = expand_scenario(client, scenario)
        run["expand_path"] = "live"
    except Exception as e:
        queries = [scenario]
        run["expand_path"] = "fallback"
        print(f"  expand fallback: {e}")
    run["search_queries"] = queries

    best = {}
    for q in queries:
        try:
            rows = backend.search(q).get("results") or []
            print(f"  search live    {len(rows):>3} rows  {q}")
        except Exception as e:
            print(f"  search failed            {q}  ({e})")
            continue
        for row in rows:
            sid = segment_id(row["source"])
            if sid not in best or row["similarity_score"] > best[sid]["similarity_score"]:
                best[sid] = row

    top = sorted(best.values(), key=lambda r: r["similarity_score"], reverse=True)[:TOP_HITS]
    hits = [build_hit(backend, client, scenario, row, rank) for rank, row in enumerate(top, 1)]
    run["hits"] = sorted(hits, key=lambda h: (h["llm"]["severity"], h["similarity"]), reverse=True)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(SCENARIOS), help="run the first N scenarios")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = parser.parse_args()

    load_env()
    weave_project = f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}"
    if weave is not None:
        weave.init(weave_project)
    else:
        print("weave not installed: ops run untraced")

    backend = Backend()
    backend.login()
    client = inference_client()

    runs = []
    for scenario in SCENARIOS[:args.limit]:
        print(f"scenario: {scenario}")
        run = mine(backend, client, scenario)
        hits = run["hits"]
        live = sum(h["path"] == "live" for h in hits)
        matches = sum(h["llm"]["is_match"] for h in hits)
        lat = [h["latency_ms"] for h in hits if h["path"] == "live"]
        median = statistics.median(lat) if lat else None
        print(f"  {len(hits)} hits ({live} live, {len(hits) - live} fallback), {matches} matches, "
              f"median LLM {median} ms")
        runs.append(run)

    with open(args.out, "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"wrote {args.out}")
    print(f"Weave project: https://wandb.ai/{weave_project}/weave")


if __name__ == "__main__":
    main()
