"""
eval_report.py — Grabbit Mining Co. / Edge Case Miner
Author: KJ (Keming Jiao)

What it does (plain English):
  The app saves every search result to results.json.
  The LLM marks each hit as a match or not, with a severity 1-5.
  A human then clicks Confirm or Reject.
  This script answers one question: "How often was the AI right, according to the human?"

Usage:
  python3 eval_report.py results.json            -> prints the report, writes metrics.md
  python3 eval_report.py results.json --export   -> also writes edge_cases.jsonl (confirmed hits only)

No libraries needed beyond Python itself.
"""
import json
import sys
from collections import Counter, defaultdict


def load_hits(path):
    with open(path) as f:
        data = json.load(f)
    hits = []
    for run in data["runs"]:
        for h in run["hits"]:
            h["query"] = run["query"]
            hits.append(h)
    return hits


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def build_report(hits):
    flagged = [h for h in hits if h["llm"]["is_match"]]
    reviewed = [h for h in flagged if h["review"] in ("confirmed", "rejected")]
    confirmed = [h for h in reviewed if h["review"] == "confirmed"]
    pending = [h for h in flagged if h["review"] == "pending"]

    lines = []
    lines.append("# Evaluation — Edge Case Miner\n")
    lines.append(
        f"**Headline:** the LLM flagged **{len(flagged)}** clips out of {len(hits)} search hits; "
        f"a human reviewed {len(reviewed)} and confirmed **{pct(len(confirmed), len(reviewed))}** "
        f"({len(pending)} still pending).\n"
    )
    lines.append("## How we evaluate\n")
    lines.append("1. Each query is expanded and run against every camera in the VAST index.")
    lines.append("2. For each hit, an LLM (W&B Inference) reads the Cosmos Reason caption and YOLO counts, "
                 "and returns is_match + severity 1-5 + a one-line reason.")
    lines.append("3. A human reviews every flagged clip and clicks Confirm or Reject.")
    lines.append("4. Precision = confirmed / reviewed. Only confirmed clips are exported to the test set.\n")

    # by query
    lines.append("## By query\n")
    lines.append("| Query | Hits | Flagged | Reviewed | Confirmed | Precision |")
    lines.append("|---|---|---|---|---|---|")
    for q in dict.fromkeys(h["query"] for h in hits):
        qh = [h for h in hits if h["query"] == q]
        qf = [h for h in qh if h["llm"]["is_match"]]
        qr = [h for h in qf if h["review"] in ("confirmed", "rejected")]
        qc = [h for h in qr if h["review"] == "confirmed"]
        lines.append(f"| {q} | {len(qh)} | {len(qf)} | {len(qr)} | {len(qc)} | {pct(len(qc), len(qr))} |")

    # by camera
    lines.append("\n## By camera\n")
    lines.append("| Camera | Location | Flagged | Confirmed |")
    lines.append("|---|---|---|---|")
    cams = defaultdict(lambda: [None, 0, 0])
    for h in flagged:
        c = cams[h["camera_id"]]
        c[0] = h["location"]
        c[1] += 1
        c[2] += h["review"] == "confirmed"
    for cam, (loc, f_, c_) in sorted(cams.items(), key=lambda x: -x[1][1]):
        lines.append(f"| {cam} | {loc} | {f_} | {c_} |")

    # severity: does the LLM's severity agree with the human?
    lines.append("\n## Does severity predict human confirmation?\n")
    lines.append("| LLM severity | Reviewed | Confirmed | Precision |")
    lines.append("|---|---|---|---|")
    for s in range(5, 0, -1):
        sr = [h for h in reviewed if h["llm"]["severity"] == s]
        sc = [h for h in sr if h["review"] == "confirmed"]
        if sr:
            lines.append(f"| {s} | {len(sr)} | {len(sc)} | {pct(len(sc), len(sr))} |")

    # top tags among confirmed
    tags = Counter(t for h in confirmed for t in h["llm"].get("tags", []))
    if tags:
        lines.append("\n## Most common tags in confirmed edge cases\n")
        lines.append(", ".join(f"{t} ({n})" for t, n in tags.most_common(8)))

    # latency
    lat = [h["latency_ms"] for h in hits if h.get("latency_ms")]
    if lat:
        lat.sort()
        lines.append(f"\n## Speed\n\nLLM scoring per clip: median {lat[len(lat)//2]} ms, "
                     f"slowest {lat[-1]} ms, over {len(lat)} clips.")
    return "\n".join(lines) + "\n", confirmed


def export(confirmed, path="edge_cases.jsonl"):
    with open(path, "w") as f:
        for h in confirmed:
            f.write(json.dumps({
                "clip_id": h["id"], "camera_id": h["camera_id"], "location": h["location"],
                "start_s": h["start_s"], "end_s": h["end_s"], "query": h["query"],
                "severity": h["llm"]["severity"], "tags": h["llm"].get("tags", []),
                "rationale": h["llm"]["rationale"], "verified_by": "human",
            }) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    report, confirmed = build_report(load_hits(sys.argv[1]))
    print(report)
    with open("metrics.md", "w") as f:
        f.write(report)
    if "--export" in sys.argv:
        print("wrote", export(confirmed))
