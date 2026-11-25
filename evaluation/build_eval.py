#!/usr/bin/env python3
"""Rebuild the ECHO Agent evaluation candidates from existing cleaned data.

No model calls, no database mutations, no label leakage into agent inputs.
Run: python evaluation/build_eval.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned"
OUT = ROOT / "evaluation" / "generated"
VERSION = "echo-agent-prelabels-v1"

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def parse_time(v):
    if not v: return None
    try: return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError: return None

def safe_visible(record, field, cutoff):
    """An absent or malformed event time is *not* proof of visibility."""
    t = parse_time(record.get(field))
    return t is not None and cutoff is not None and t <= cutoff

def recommendations(scene):
    if any(x in scene for x in ["不良反应", "过敏", "孕妇", "使用不适"]):
        return ("health_safety", ["product_identity", "symptom_statement", "verified_product_information"],
                ["guaranteed_safe", "medical_diagnosis", "guaranteed_compensation"], "human_review_priority")
    if any(x in scene for x in ["退款", "退货", "价保", "保价", "补差", "报销"]):
        return ("refund_after_sales", ["order_identity", "verified_refund_or_policy_status"],
                ["refund_completed_without_proof", "guaranteed_arrival_time"], "review_if_status_unknown")
    if any(x in scene for x in ["物流", "快递", "发货", "少件", "破损", "补发", "换货", "改址", "赠品"]):
        return ("fulfillment_after_sales", ["order_identity", "shipment_or_ticket_status"],
                ["shipment_confirmed_without_proof", "address_changed_without_approval"], "review_if_write_needed")
    if "产品咨询" in scene:
        return ("product_consultation", ["verified_product_knowledge"],
                ["unsupported_product_claims"], "draft_only")
    return ("general_service", ["conversation_context"], ["unsupported_factual_claims"], "draft_only")

def main():
    messages = read_jsonl(DATA / "chat_messages.jsonl")
    orders = read_jsonl(DATA / "orders.jsonl")
    tickets = read_jsonl(DATA / "tickets.jsonl")
    by_session = defaultdict(list)
    for m in messages:
        by_session[str(m["session_id"])].append(m)
    for ms in by_session.values():
        ms.sort(key=lambda x: (int(x.get("message_seq") or 0), str(x.get("message_id") or "")))

    # Two distinct original sessions per observed scene, sorted deterministically.
    by_scene = defaultdict(list)
    for sid in sorted(by_session):
        first = next((m for m in by_session[sid] if m.get("is_buyer_message")), by_session[sid][0])
        scene = (str(first.get("scene_major") or "") + " / " + str(first.get("scene_minor") or ""))
        by_scene[scene].append(sid)
    selected = [(scene, sid) for scene, ids in by_scene.items() for sid in ids[:2]]
    by_order = defaultdict(list); by_ticket = defaultdict(list)
    for o in orders: by_order[str(o.get("session_id"))].append(o)
    for t in tickets: by_ticket[str(t.get("session_id"))].append(t)

    candidates, nodes, review = [], [], []
    for scene, sid in selected:
        ms = by_session[sid]
        buyers = [m for m in ms if m.get("is_buyer_message")]
        candidates.append({"session_id":sid,"scene":scene,"message_count":len(ms),"buyer_checkpoints":len(buyers),
                           "order_count":len(by_order[sid]),"ticket_count":len(by_ticket[sid]),
                           "review_status":"unreviewed"})
        for m in buyers:
            cutoff = parse_time(m.get("sent_at"))
            seq = int(m.get("message_seq") or 0)
            visible_order = [o for o in by_order[sid] if safe_visible(o, "ordered_at", cutoff)]
            visible_ticket = [t for t in by_ticket[sid] if safe_visible(t, "created_at", cutoff)]
            ambiguous_order = [o for o in by_order[sid] if parse_time(o.get("ordered_at")) is None]
            ambiguous_ticket = [t for t in by_ticket[sid] if parse_time(t.get("created_at")) is None]
            intent, facts, forbidden, outcome = recommendations(scene)
            unknown = ["historical_order_status_at_cutoff", "historical_ticket_status_at_cutoff",
                       "refund_and_tracking_live_state", "effective_policy_and_product_knowledge_at_cutoff"]
            needs_high = any(s in scene for s in ["不良反应", "过敏", "孕妇", "仅退款疑似", "拦截改址", "质疑正品"])
            node_id = f"{sid}-M{seq:03d}"
            nodes.append({
                "case_id":node_id, "session_id":sid,
                "as_of_message_id":str(m.get("message_id") or ""),
                "as_of_message_seq":seq, "as_of_timestamp":m.get("sent_at"),
                "input_ref":{"file":"data/cleaned/chat_messages.jsonl","message_id":m.get("message_id"),
                             "context_rule":"same session messages with seq <= cutoff; other data time-filtered"},
                "scene_from_dataset":scene,
                "prelabels":{"intent_family":intent,"intent_source":"scene_metadata_proxy",
                             "task_types":["retrieve_facts","draft_reply"] if visible_order else ["draft_reply"],
                             "fact_requirements_suggested":facts,"forbidden_claims_suggested":forbidden,
                             "acceptable_outcome_suggested":outcome},
                "available_fact_refs_at_cutoff":{"order_nos":[str(o["order_no"]) for o in visible_order],
                    "ticket_nos":[str(t["ticket_no"]) for t in visible_ticket]},
                "unverified_temporal_facts":unknown,
                "records_with_missing_event_time":{"orders":len(ambiguous_order),"tickets":len(ambiguous_ticket)},
                "governance":{"agent_may_auto_send":False,"write_action_requires_human_approval":True},
                "review":{"status":"unreviewed","priority":"high" if needs_high else "normal",
                          "required":["ground_truth_intent","necessary_evidence","acceptable_result",
                                      "policy_version_or_unknown","time_consistency"]},
                "label_status":"auto_proposed_not_gold","dataset_version":VERSION
            })
            review.append({"case_id":node_id,"session_id":sid,"scene":scene,
                           "message_seq":seq,"priority":"high" if needs_high else "normal",
                           "missing_annotations":"gold_intent;necessary_evidence;acceptable_result;risk_level;policy_version;time_audit",
                           "temporal_limitations":"order/ticket historical state unavailable; check timestamp completeness",
                           "reviewer":"","decision":"pending"})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("candidates.jsonl", candidates),("replay_nodes.jsonl",nodes)]:
        with (OUT/name).open("w",encoding="utf-8") as f:
            for x in data: f.write(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n")
    with (OUT/"review_queue.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    stats={"dataset_version":VERSION,"source_totals":{"messages":len(messages),"sessions":len(by_session),
            "orders":len(orders),"tickets":len(tickets)},"selected_sessions":len(candidates),
           "observed_scenes":len(by_scene),"buyer_message_nodes":len(nodes),
           "high_priority_nodes":sum(x["review"]["priority"]=="high" for x in nodes),
           "label_status":"auto_proposed_not_gold",
           "limitations":["do not use future messages as agent input",
              "as-of timestamps on order/ticket creation do not reconstruct historical status",
              "historical policy and product knowledge not versioned in these data",
              "original scene label is not gold intent and may be repetitive",
              "do not split different nodes of a session across train/test"]}
    (OUT/"summary.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
