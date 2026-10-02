# Evaluation — Edge Case Miner

**Headline:** the LLM flagged **17** clips out of 36 search hits; a human reviewed 15 and confirmed **60%** (2 still pending).

## How we evaluate

1. Each query is expanded and run against every camera in the VAST index.
2. For each hit, an LLM (W&B Inference) reads the Cosmos Reason caption and YOLO counts, and returns is_match + severity 1-5 + a one-line reason.
3. A human reviews every flagged clip and clicks Confirm or Reject.
4. Precision = confirmed / reviewed. Only confirmed clips are exported to the test set.

## By query

| Query | Hits | Flagged | Reviewed | Confirmed | Precision |
|---|---|---|---|---|---|
| forklift within 2 m of a worker in an aisle | 12 | 10 | 10 | 7 | 70% |
| person close to a moving vehicle | 12 | 5 | 5 | 2 | 40% |
| vehicle braking hard | 12 | 2 | 0 | 0 | n/a |

## By camera

| Camera | Location | Flagged | Confirmed |
|---|---|---|---|
| sdg_warehouse_cam-2 | warehouse3 | 13 | 9 |
| neighborhood_cam-1 | neighborhood | 2 | 0 |
| pie_cam-3 | toronto | 2 | 0 |

## Does severity predict human confirmation?

| LLM severity | Reviewed | Confirmed | Precision |
|---|---|---|---|
| 5 | 2 | 2 | 100% |
| 4 | 2 | 0 | 0% |
| 3 | 4 | 4 | 100% |
| 2 | 7 | 3 | 43% |

## Most common tags in confirmed edge cases

forklift (6), worker (5), collision hazard (4), aisle (4), high risk (2), moving vehicle (2), worker in aisle (1), forklift safety (1)

## Speed

LLM scoring per clip: median 402 ms, slowest 587 ms, over 36 clips.
