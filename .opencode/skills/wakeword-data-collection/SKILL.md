---
name: wakeword-data-collection
description: Use when a user needs to record wake-word training/validation/test audio, asks how many samples to record, what negatives to collect, how to use the phone recorder app or desktop record command, or how to pull recordings off the phone (fetch-recordings, adb). Covers split discipline and split-integrity rules.
---

# Wake-word data collection

Real recordings beat synthetic data. **Phone-mic recordings beat desktop recordings** — the deployed model runs on the phone, and its mic/AGC/noise-suppression shift scores measurably (observed: 0.56 on phone vs 0.58 calibrated threshold from desktop takes — a miss).

## How much to record

| Data | Minimum | Good | Where |
|---|---|---|---|
| Positive train | 20 takes | 50+, multiple rooms/distances | train |
| Positive validation | 10 | 20, separate session | validation |
| Positive test | 10 | 20, separate session, used ONCE | test |
| Negative train | 20 takes | near-misses ("Hey Box"), random speech, TV/ambient | train |
| Negative validation | 10 | real ambient + speech | validation |

Vary: distance from mic (close / across the room), loudness, room, background conditions. Other people saying the wake word help more than more of the same voice. Different sessions = different split groups (the tooling does this automatically).

## Recording tools

**Phone (preferred):**

```bash
wakeword-lab build-recorder <proj> --install   # builds android/recorder, wake word on top
# 6 push-to-talk buttons: POS/NEG x train/validation/test
# hold = record, release = save, slide off = cancel
wakeword-lab fetch-recordings <proj> --speaker <name> [--delete]   # adb pull into data/raw
```

**Desktop (needs `pip install -e '.[record]'`):**

```bash
wakeword-lab record <proj> --speaker <name> [--kind positive|negative|noise] [--split train|validation|test]
```

## Split integrity — the rules that keep metrics honest

- **Never move files between folders or re-balance by hand.** Assign splits via `fetch-recordings`/`record --split` or the `split` column in `data/raw/metadata.csv`, then re-run `prepare`.
- Split groups (speaker/session) must not straddle partitions — one group's clips all land in the same split. The tooling pins `group_id` per recording session for this reason.
- Noise mixing uses each split's own pinned noise groups; never cross-mix.
- The test split is a **one-time final check**. If it has been evaluated repeatedly while iterating, record a fresh test session for the final number.

## After every data change

```bash
wakeword-lab prepare <proj>   # regenerates data/processed; experiment skips prepare if data is stale
```
