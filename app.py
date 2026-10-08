from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.database import (
    DEFAULT_DB_PATH,
    create_agent_run,
    create_reply_feedback,
    create_database,
    create_risk_event,
    get_case,
    list_session_ids,
    update_risk_event,
)


ROOT = Path(__file__).parent
STATIC = ROOT / "static"
DB_PATH = DEFAULT_DB_PATH
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# ThreadingHTTPServer 可以同时处理多个请求，但本地 Qdrant 使用文件锁，模型推理
# 也会占用较多内存。第一版先串行运行分析请求，保证 Demo 稳定。
PIPELINE_LOCK = threading.Lock()


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _shorten(value: str, length: int = 58) -> str:
    value = _text(value).replace("\n", " ")
    return value if len(value) <= length else value[: length - 1] + "…"


def _time_label(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%m-%d %H:%M")
    except ValueError:
        return text[:16]


def _is_buyer(message: dict) -> bool:
    role = _text(message.get("role"))
    return "买家" in role or message.get("is_target_buyer_message") == 1


def _ticket_detail(ticket: dict) -> dict:
    try:
        return json.loads(ticket.get("detail_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _case_risk(case: dict) -> tuple[str, str, str, str, int]:
    messages = case.get("messages", [])
    tickets = case.get("tickets", [])
    text = " ".join(_text(item.get("message_text")) for item in messages)
    details = [_ticket_detail(item) for item in tickets]
    combined = text + " " + " ".join(json.dumps(item, ensure_ascii=False) for item in details)

    if re.search(r"不良反应|过敏|泛红|刺痒|起疹|就医", combined):
        return "高风险", "red", "用户描述了可能的不良反应或就医相关情况，建议暂停继续推荐并转人工专人处理。", "担忧，需要人工关注", 88
    if any(_text(detail.get("是否异常")) == "是" for detail in details) or "重复退款" in combined:
        return "高风险", "red", "售后记录存在异常标记或退款核实需求，暂不应向用户承诺未经确认的退款结果。", "不满，需要明确进度", 78
    if tickets and any(_text(item.get("status")) not in {"完成", "已完成", "已关闭"} for item in tickets):
        return "中风险", "amber", "相关服务工单尚未完结，建议明确当前处理状态和下一步时间预期，减少重复追问。", "焦虑上升，需要关注", 66
    if re.search(r"投诉|生气|为什么还|说过很多遍|不满意", combined):
        return "中风险", "amber", "对话中出现明显不满或重复追问，建议先回应情绪，再说明可确认的处理进度。", "不满，需要关注", 62
    return "低风险", "blue", "当前未命中高风险规则，仍需结合具体业务状态处理。", "平稳", 32


def _present_case(case: dict) -> dict:
    session = case["session"]
    messages = case.get("messages", [])
    orders = case.get("orders", [])
    tickets = case.get("tickets", [])
    timeline = case.get("timeline", [])
    risk, risk_tone, risk_reason, emotion, emotion_score = _case_risk(case)
    latest_scene = next((_text(item.get("scene_major")) for item in reversed(messages) if _text(item.get("scene_major"))), "其他服务")
    issue = next((_text(item.get("scene_minor")) for item in reversed(messages) if _text(item.get("scene_minor"))), _text(tickets[0].get("ticket_type")) if tickets else "待识别问题")
    order = orders[0] if orders else {}
    ticket = tickets[0] if tickets else {}
    buyer = _text(session.get("buyer_nickname")) or "匿名用户"

    frontend_messages = [
        {
            "message_id": _text(item.get("message_id")),
            "message_seq": item.get("message_seq"),
            "role": "buyer" if _is_buyer(item) else "agent",
            "time": _time_label(item.get("sent_at")),
            "text": _text(item.get("message_text")) or _text(item.get("chat_content")) or "（非文本消息）",
        }
        for item in messages
    ]
    frontend_timeline = []
    for item in timeline:
        try:
            raw = json.loads(item.get("detail_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            raw = {}
        if item.get("event_type") == "chat":
            detail = _text(raw.get("message_text")) or _text(raw.get("chat_content")) or "聊天事件"
        elif item.get("event_type") == "order":
            detail = " · ".join(part for part in [_text(raw.get("商品名称")), _text(raw.get("订单状态"))] if part)
        else:
            detail = " · ".join(part for part in [_text(item.get("status")), _text(raw.get("售后原因"))] if part)
        frontend_timeline.append({"time": _time_label(item.get("event_time")), "title": _text(item.get("title")) or "服务事件", "detail": _shorten(detail or "暂无详细描述", 100), "tone": "red" if risk_tone == "red" else "amber" if risk_tone == "amber" else "blue"})

    evidence = []
    if ticket.get("ticket_no"):
        evidence.append(f"{ticket.get('ticket_type')}工单 {ticket.get('ticket_no')}")
    if ticket.get("status"):
        evidence.append(f"处理状态：{ticket.get('status')}")
    if order.get("order_no"):
        evidence.append(f"订单 {order.get('order_no')}")
    if not evidence:
        evidence.append("当前会话记录")
    reply = f"抱歉让您遇到这个问题。我已经查看了当前会话和相关服务记录，目前识别为“{issue}”，{risk_reason}我会先为您记录并跟进可确认的处理进度，后续结果会及时同步给您。"
    return {
        "id": _text(session.get("session_id")), "name": buyer, "scene": latest_scene, "issue": issue,
        "risk": risk, "riskTone": risk_tone, "riskReason": risk_reason, "emotion": emotion, "emotionScore": emotion_score,
        "order": _text(order.get("product_name")) or "暂无关联订单", "orderStatus": _text(order.get("order_status")) or "暂无订单状态",
        "orderAmount": f"¥{order.get('paid_amount')}" if order.get("paid_amount") is not None else "—", "orderNo": _text(order.get("order_no")),
        "ticket": f"{ticket.get('ticket_type')}工单 {ticket.get('ticket_no')}" if ticket else "暂无关联工单", "ticketStatus": _text(ticket.get("status")) or "—",
        "lastSeen": _time_label(session.get("last_message_at")) or "暂无", "messages": frontend_messages, "timeline": frontend_timeline,
        "evidence": evidence, "reply": reply, "facts": {"session": session, "orders": orders, "tickets": tickets},
    }


def list_cases(limit: int = 20) -> list[dict]:
    return [_present_case(get_case(DB_PATH, session_id)) for session_id in list_session_ids(DB_PATH, limit)]


def list_risk_events() -> list[dict]:
    connection = create_database(DB_PATH)
    try:
        rows = connection.execute("SELECT * FROM risk_events ORDER BY CASE risk_level WHEN '高风险' THEN 1 WHEN '中风险' THEN 2 ELSE 3 END, updated_at DESC").fetchall()
        if rows:
            return [{"id": f"R-{row['id']:04d}", "customer": row["buyer_nickname"] or row["session_id"], "issue": row["risk_type"], "reason": row["reason"], "level": row["risk_level"], "tone": "red" if row["risk_level"] == "高风险" else "amber", "owner": row["owner"] or "待分派", "status": row["status"], "session_id": row["session_id"]} for row in rows]
    finally:
        connection.close()
    return [{"id": f"AUTO-{item['id']}", "customer": item["name"], "issue": f"{item['scene']} · {item['issue']}", "reason": item["riskReason"], "level": item["risk"], "tone": item["riskTone"], "owner": "待分派", "status": "待处理", "session_id": item["id"]} for item in list_cases(200) if item["risk"] != "低风险"]


def _default_query(case: dict) -> str:
    """未显式输入问题时，使用当前会话最后一条消费者消息作为 query。"""

    for message in reversed(case.get("messages", [])):
        if _is_buyer(message):
            text = _text(message.get("message_text"))
            if text:
                return text
    return "请分析当前消费者会话"


def run_case_copilot(payload: dict) -> dict:
    """把 Web 请求转换为统一流水线参数。"""

    # 延迟导入：浏览普通列表时不需要提前加载 AI 相关依赖。
    from run_copilot_pipeline import run_copilot_pipeline

    session_id = _text(payload.get("session_id"))
    if not session_id:
        raise ValueError("缺少 session_id")
    case = get_case(DB_PATH, session_id)
    query = _text(payload.get("query")) or _default_query(case)

    as_of_message_seq = payload.get("as_of_message_seq")
    if as_of_message_seq not in (None, ""):
        as_of_message_seq = int(as_of_message_seq)
    else:
        as_of_message_seq = None

    with PIPELINE_LOCK:
        result = run_copilot_pipeline(
            session_id=session_id,
            query=query,
            db_path=DB_PATH,
            as_of_message_id=(
                _text(payload.get("as_of_message_id")) or None
            ),
            as_of_message_seq=as_of_message_seq,
            skip_llm=bool(payload.get("skip_llm", False)),
        )

    llm_status = (
        result.get("analysis", {})
        .get("raw_analysis", {})
        .get("llm", {})
        .get("status", "unknown")
    )
    trace_id = uuid.uuid4().hex
    run_status = "success" if llm_status == "success" else "degraded"
    agent_run_id = create_agent_run(
        DB_PATH,
        session_id=session_id,
        buyer_nickname=_text(case.get("session", {}).get("buyer_nickname")) or None,
        trace_id=trace_id,
        status=run_status,
        intent=result.get("analysis", {}).get("final", {}).get("intent", {}),
        retrieval=result.get("context", {}).get("retrieval", {}),
        reply=result.get("reply", {}).get("suggested_reply", {}),
        # 当前 urllib 调用还没有返回 token usage，先保留阶段耗时；后续再扩展。
        token_usage={
            "stage_timings_ms": result.get("observability", {}).get(
                "stage_timings_ms", {}
            )
        },
        error_message=(
            result.get("analysis", {})
            .get("raw_analysis", {})
            .get("llm", {})
            .get("error")
        ),
    )
    result.setdefault("observability", {})["trace_id"] = trace_id
    result["observability"]["agent_run_id"] = agent_run_id
    return result


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)
        if path == "/api/health":
            self._json({"status": "ok", "service": "echo-beauty-copilot", "database": DB_PATH.exists()})
            return
        if path == "/api/cases":
            self._json(list_cases(int(query.get("limit", [20])[0])))
            return
        if path.startswith("/api/cases/"):
            session_id = path.rsplit("/", 1)[-1]
            try:
                self._json(_present_case(get_case(DB_PATH, session_id)))
            except KeyError:
                self._json({"error": "会话不存在", "session_id": session_id}, status=404)
            return
        if path == "/api/risk-events":
            self._json(list_risk_events())
            return
        if path == "/api/stats":
            connection = create_database(DB_PATH)
            try:
                self._json({"buyers": connection.execute("SELECT COUNT(*) FROM buyers").fetchone()[0], "sessions": connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], "messages": connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], "orders": connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], "tickets": connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0], "risk_events": len(list_risk_events())})
            finally:
                connection.close()
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024 * 1024:
                raise ValueError("请求体不能超过 1 MB")
            payload = json.loads(self.rfile.read(length) or "{}")

            if path == "/api/copilot/analyze":
                self._json(run_case_copilot(payload))
                return

            if path == "/api/reply-feedback":
                session_id = _text(payload.get("session_id"))
                action = _text(payload.get("action"))
                original_reply = _text(payload.get("original_reply"))
                if not session_id:
                    raise ValueError("缺少 session_id")
                feedback_id = create_reply_feedback(
                    DB_PATH,
                    session_id=session_id,
                    action=action,
                    original_reply=original_reply,
                    final_reply=_text(payload.get("final_reply")) or None,
                    reason=_text(payload.get("reason")) or None,
                    analysis_schema_version=(
                        _text(payload.get("analysis_schema_version")) or None
                    ),
                )
                self._json({"id": feedback_id, "status": "created"}, status=201)
                return

            if path != "/api/risk-events":
                self._json({"error": "接口不存在"}, status=404)
                return

            session_id = _text(payload.get("session_id"))
            if not session_id:
                raise ValueError("缺少 session_id")
            case = _present_case(get_case(DB_PATH, session_id))
            event_id = create_risk_event(DB_PATH, session_id=session_id, buyer_nickname=case["name"], risk_type=payload.get("risk_type") or f"{case['scene']} · {case['issue']}", risk_level=payload.get("risk_level") or case["risk"], reason=payload.get("reason") or case["riskReason"], evidence=payload.get("evidence") or case["evidence"], owner=payload.get("owner"))
            self._json({"id": event_id, "status": "created"}, status=201)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, status=400)
        except Exception as error:
            # 模型或向量库错误返回可读信息，避免 HTTP 线程直接中断。
            self._json(
                {"error": "客服辅助分析失败", "detail": str(error)},
                status=500,
            )

    def do_PATCH(self):
        """更新风险事件状态；该操作由人工客服或主管明确触发。"""

        path = urlparse(self.path).path.rstrip("/")
        if not path.startswith("/api/risk-events/"):
            self._json({"error": "接口不存在"}, status=404)
            return
        try:
            raw_id = path.rsplit("/", 1)[-1]
            event_id = int(raw_id.removeprefix("R-"))
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or "{}")
            updated = update_risk_event(
                DB_PATH,
                event_id=event_id,
                status=_text(payload.get("status")),
                owner=_text(payload.get("owner")) or None,
            )
            if not updated:
                self._json({"error": "风险事件不存在"}, status=404)
                return
            self._json({"id": event_id, "status": "updated"})
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, status=400)

    def _json(self, payload: object, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(os.environ.get("ECHO_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"ECHO Beauty running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ECHO Beauty")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
