# Grabbit Mining Co. 🐰⛏️

**Describe a dangerous moment in one sentence. Get a human-verified edge-case test set from video nobody has watched.**

> Headline: across **[TOTAL_SEGMENTS]** indexed video segments, our agent flagged **[FLAGGED]** risky clips; a human reviewed them and confirmed **[PRECISION]**. Median LLM scoring time: **[MEDIAN_MS] ms** per clip. *(numbers from `eval_report.py`, Oct 2 2026)*

Live app: [APP_URL] · Demo video: [VIDEO_URL] · Team: LJ (build) & KJ (product, evaluation, human review)

---

## The problem
Robotics and self-driving teams need *edge cases* — the near-miss where a forklift turns toward a worker, or a truck brakes hard after a cut-in. Those moments are buried in hours of footage nobody watches. Today, finding them means a person scrubbing video for days.

## What we built
A video agent on top of the VAST video search & summary stack:

| Step | What happens | Sponsor tool |
|---|---|---|
| 01 Expand | One plain-English scenario becomes 3–5 search queries | W&B Inference |
| 02 Search | Every camera in the index is searched (highway, warehouse, streets) | VAST DataEngine + VastDB, NVIDIA Cosmos Embed |
| 03 Evidence | Each hit carries a Cosmos Reason description (re-ingested with our risk prompt) and YOLO object counts | NVIDIA Cosmos Reason, YOLO11 on CoreWeave GPUs |
| 04 Score | An LLM returns match / severity 1–5 / one-line reason | W&B Inference, traced in W&B Weave |
| 05 Human review | Grabbit can't reach its own button — a human clicks Confirm or Reject | — |
| 06 Act | Confirmed clips are appended to `edge_cases.jsonl`, a regression test set | — |

Built with Cursor on the VAST workshop VM.

## Inputs and outputs
- **Input:** a scenario sentence, e.g. `person close to a moving vehicle`.
- **Intermediate:** `results.json` — every hit with camera, timestamps, caption, detections, LLM judgment, human review, latency. Schema: see `sample_results.json`.
- **Output:** `edge_cases.jsonl` — human-confirmed clips only, one per line: clip id, camera, start/end seconds, severity, tags, rationale, `verified_by: human`.

## How we evaluate
1. Run the three demo queries against the full team index.
2. The LLM flags matches and assigns severity.
3. A human reviews every flagged clip (Confirm / Reject).
4. `eval_report.py` computes precision = confirmed / reviewed, broken down by query, by camera, and by LLM severity (does a higher severity really mean the human agrees more often?).

```bash
python3 eval_report.py results.json --export   # prints report, writes metrics.md and edge_cases.jsonl
```

Results: see [`metrics.md`](metrics.md).

## Re-ingestion prompt
Cosmos Reason only writes down what the prompt asks for, so we re-ingested warehouse (Pack C) and highway (Pack A) clips with:

> Describe every person and every vehicle/forklift in the segment. For each person, state the approximate distance to the nearest moving vehicle (touching / under 2m / 2-5m / far), whether the vehicle is moving, braking, or turning, and whether the person is in a walkway or occluded. For vehicles, note hard braking, sudden lane changes, or tailgating. End with: RISK = low/medium/high.

## Reproduce
1. On the VAST workshop VM: `cd ~/vast-builders-challenge && git pull`, then clone this repo next to it.
2. Credentials and endpoints come from the team environment (`/config/<team>.config`); nothing is hard-coded.
3. Start the app: [RUN_COMMAND]. Open `/app` on the team host.
4. Replay mode without the backend: open the app with `?demo=1` (reads cached `results.json`, labeled "Replay" on screen).

## Honest limitations
- Only clips we re-ingested carry distance/risk descriptions; the rest use the default captions.
- Precision is measured on [REVIEWED] human-reviewed clips by one reviewer — a small sample.
- Severity is an LLM judgment, not a physical measurement of distance.

## Next
Feed `edge_cases.jsonl` straight into real-world robot policy evaluation — the data layer we wished we had.
