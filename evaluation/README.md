# ECHO Beauty Agent Evaluation v1

## Dataset status

**Auto-generated candidate benchmark, NOT a human-reviewed gold benchmark.**

- Source: `data/cleaned/chat_messages.jsonl`, `orders.jsonl`, `tickets.jsonl` from this repository.
- 82 real source sessions: first two distinct sessions for each of 41 observed scene labels.
- 294 buyer-message decision checkpoints; assistant/cs messages are context only.
- 68 high-priority review checkpoints according to **provisional** scenario heuristics.
- No raw messages are copied into generated node files. Each node is a pointer to source `session_id` + `message_id`.

## Files

| File | Purpose |
|---|---|
| `build_eval.py` | Deterministically rebuild all generated files from existing cleaned data |
| `generated/candidates.jsonl` | 82 unique selected sessions, scene and source counts |
| `generated/replay_nodes.jsonl` | 294 message-level replay checkpoints and **suggested** labels |
| `generated/review_queue.csv` | Spreadsheet-friendly human-review checklist |
| `generated/summary.json` | Selection statistics and known limitations |

Run from repo root:

```bash
python evaluation/build_eval.py
```

## How to use a replay node

For every node:

1. Load the source session identified by `session_id`.
2. Include **only messages with `message_seq <= as_of_message_seq`**.
3. Include an order only if its creation/order time is known and `ordered_at <= as_of_timestamp`. Include a ticket only if its creation time is known and `created_at <= as_of_timestamp`.
4. Do **not** use the final `order_status` / `ticket.status` as historical state; status changes are not time-versioned in these exports.
5. Load policies and product knowledge only if they can be proven valid as of the checkpoint; otherwise label as unknown.
6. Agent receives the **input context** only, not `prelabels` or `review`. Evaluator may inspect labels after execution.
7. Use earlier客服 messages only as observed historical dialogue; **do not use later客服 messages as answers**.

**Important:** `prelabels.intent_family` comes from dataset scene metadata rather than independent annotation. It is a weak label, not a verified intent prediction.

## What still requires human review

- Gold intent and task boundary for each decision checkpoint
- Mandatory factual evidence and whether evidence is sufficient, not just present
- Acceptable answer and prohibited claims for each specific case
- Risk level and escalation decision, particularly adverse reactions, pregnancy, fraud suspicion, address changes, and payment
- Historical policy versions, business state transitions and temporal discrepancies
- Independent review of duplicates/template similarity, source privacy, and leakage

`review_queue.csv` contains pending entries. Review 20-30 diverse cases first, then expand.

## Selection and evaluation warnings

- 82 source sessions are **not** 294 independent cases.
- Do not split checkpoints of the same session across train/dev/test. Near-duplicate scene templates should also be grouped where appropriate.
- Many source sessions have only one order and one ticket; multi-order disambiguation and tool-failure scenarios require **separate synthetic fixtures**.
- The data files may contain consumer details. Confirm provenance, authorization, masking and access controls before public release.
- No Agent execution or outcome scoring has been performed for this benchmark yet.
