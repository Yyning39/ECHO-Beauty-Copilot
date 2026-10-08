"""
1. 根据当前 session_id 查询当前服务会话
2. 查询当前会话中的聊天、订单、工单和时间线
3. 根据混合检索结果查询历史服务会话
4. 将检索结果和 SQLite 精确事实组合起来
5. 输出适合后续 LLM 使用的上下文
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)
from backend.database import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_case,
)
def read_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def parse_datetime(value: Any) -> datetime | None:
    """把数据中的时间字符串转换为可比较的 datetime。"""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        # 一条异常时间不应该让整条客服分析流程失败。
        return None


def resolve_analysis_cutoff(
    case: dict[str, Any],
    as_of_message_id: str | None = None,
    as_of_message_seq: int | None = None,
) -> dict[str, Any] | None:
    """找到本次离线回放允许看到的最后一条消息。

    如果分析第 1 条消息时读取了后面的回复，就发生了“未来信息泄露”。
    截断点使离线测试与真实客服当时能看到的信息保持一致。
    """

    if as_of_message_id is None and as_of_message_seq is None:
        return None

    for message in case.get("messages", []):
        message_id = str(message.get("message_id") or "")
        message_seq = message.get("message_seq")
        id_matched = (
            as_of_message_id is not None
            and message_id == str(as_of_message_id)
        )
        seq_matched = (
            as_of_message_seq is not None
            and message_seq is not None
            and int(message_seq) == int(as_of_message_seq)
        )
        if id_matched or seq_matched:
            return {
                "message_id": message_id,
                "message_seq": message_seq,
                "sent_at": message.get("sent_at"),
            }

    selector = (
        f"message_id={as_of_message_id}"
        if as_of_message_id is not None
        else f"message_seq={as_of_message_seq}"
    )
    raise ValueError(
        f"当前会话中找不到截断消息：{selector}"
    )


def filter_case_as_of(
    case: dict[str, Any],
    cutoff: dict[str, Any] | None,
) -> dict[str, Any]:
    """过滤截断点之后的消息、订单、工单和时间线事件。"""

    if cutoff is None:
        return case

    cutoff_seq = cutoff.get("message_seq")
    cutoff_time = parse_datetime(cutoff.get("sent_at"))

    visible_messages: list[dict[str, Any]] = []
    for message in case.get("messages", []):
        message_seq = message.get("message_seq")
        message_time = parse_datetime(message.get("sent_at"))
        if cutoff_seq is not None and message_seq is not None:
            visible = int(message_seq) <= int(cutoff_seq)
        elif cutoff_time is not None and message_time is not None:
            visible = message_time <= cutoff_time
        else:
            visible = True
        if visible:
            # 创建新字典，避免清理关系字段时修改数据库返回的原对象。
            visible_messages.append(dict(message))

    def happened_before_cutoff(
        record: dict[str, Any],
        time_field: str,
    ) -> bool:
        record_time = parse_datetime(record.get(time_field))
        return (
            cutoff_time is None
            or record_time is None
            or record_time <= cutoff_time
        )

    visible_orders = [
        order
        for order in case.get("orders", [])
        if happened_before_cutoff(order, "ordered_at")
    ]
    visible_tickets = [
        ticket
        for ticket in case.get("tickets", [])
        if happened_before_cutoff(ticket, "created_at")
    ]
    visible_timeline = [
        event
        for event in case.get("timeline", [])
        if happened_before_cutoff(event, "event_time")
    ]

    # 赛事数据可能把后来创建的工单号回填到更早的消息中。回放到工单创建
    # 之前时，需要移除这个未来关系；否则 order_binding 也会提前知道答案。
    all_ticket_nos = {
        str(ticket.get("ticket_no"))
        for ticket in case.get("tickets", [])
        if ticket.get("ticket_no")
    }
    visible_ticket_nos = {
        str(ticket.get("ticket_no"))
        for ticket in visible_tickets
        if ticket.get("ticket_no")
    }
    for message in visible_messages:
        related_ticket_no = message.get("related_ticket_no")
        if (
            related_ticket_no
            and str(related_ticket_no) in all_ticket_nos
            and str(related_ticket_no) not in visible_ticket_nos
        ):
            message["related_ticket_no"] = None

    filtered_session = dict(case.get("session", {}))
    if cutoff.get("sent_at"):
        filtered_session["last_message_at"] = cutoff.get("sent_at")

    return {
        **case,
        "session": filtered_session,
        "messages": visible_messages,
        "orders": visible_orders,
        "tickets": visible_tickets,
        "timeline": visible_timeline,
    }

def load_history_hits(
    path: Path | None,
) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = read_json_file(path)
    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]
    if isinstance(data, dict):
        results = data.get(
            "hybrid_results",
            [],
        )
        if not isinstance(results, list):
            return []
        return [
            item
            for item in results
            if isinstance(item, dict)
        ]
    raise ValueError(
        "历史检索结果必须是列表或字典"
    )

def parse_json_field(
    value: Any,
    default: Any,
) -> Any:
    """
    解析 SQLite 中保存的 JSON 字段。
    例如：detail_json = '{"症状": "泛红"}'
    转换成：{"症状": "泛红"}
    """

    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def compact_case(
    case: dict[str, Any],
    include_messages: bool = True,
) -> dict[str, Any]:
    """
    这里进行一次字段筛选，
    只保留当前 Agent 需要的内容。
    """
    session = case["session"]
    result: dict[str, Any] = {
        "session": {
            "session_id": session.get(
                "session_id"
            ),
            "buyer_nickname": session.get(
                "buyer_nickname"
            ),
            "store": session.get(
                "store"
            ),
            "first_message_at": session.get(
                "first_message_at"
            ),
            "last_message_at": session.get(
                "last_message_at"
            ),
        },
        "orders": [],
        "tickets": [],
        "timeline": [],
    }

    if include_messages:
        result["messages"] = []
        for message in case.get(
            "messages",
            [],
        ):
            result["messages"].append(
                {
                    "message_id": message.get(
                        "message_id"
                    ),
                    "message_seq": message.get(
                        "message_seq"
                    ),
                    "sent_at": message.get(
                        "sent_at"
                    ),
                    "role": message.get(
                        "role"
                    ),
                    "message_text": message.get(
                        "message_text"
                    ),
                    # get() 从消息字典中读取字段；字段缺失时返回 None。
                    # 订单关联来自 Excel 的“关联订单号”，供 order_binding 核对。
                    # 数据关联不代表消费者已经确认过这笔订单。
                    "related_order_no": message.get(
                        "related_order_no"
                    ),
                    # 保留“关联工单号”，后续可通过工单找到对应订单。
                    "related_ticket_no": message.get(
                        "related_ticket_no"
                    ),
                    # 保留消息形式；赛事数据目前包含“文本”和“图片”。
                    "content_type": message.get(
                        "content_type"
                    ),
                    # 保留 Excel 的原始内容字符串，不在这里假定存在订单卡片。
                    # 如需解析其中的 JSON，由后续使用该字段的模块处理。
                    "chat_content": message.get(
                        "chat_content"
                    ),
                    # 图片消息保留图片路径，方便后续读取图片；路径本身不是图片分析结果。
                    "image_path": message.get(
                        "image_path"
                    ),
                }
            )
    for order in case.get(
        "orders",
        [],
    ):
        result["orders"].append(
            {
                "order_no": order.get(
                    "order_no"
                ),
                "product_name": order.get(
                    "product_name"
                ),
                "quantity": order.get(
                    "quantity"
                ),
                "paid_amount": order.get(
                    "paid_amount"
                ),
                "order_status": order.get(
                    "order_status"
                ),
                "ordered_at": order.get(
                    "ordered_at"
                ),
                "paid_at": order.get(
                    "paid_at"
                ),
                "shipped_at": order.get(
                    "shipped_at"
                ),
                "carrier": order.get(
                    "carrier"
                ),
                "tracking_no": order.get(
                    "tracking_no"
                ),
            }
        )

    for ticket in case.get(
        "tickets",
        [],
    ):
        result["tickets"].append(
            {
                "ticket_no": ticket.get(
                    "ticket_no"
                ),
                "ticket_type": ticket.get(
                    "ticket_type"
                ),
                "order_no": ticket.get(
                    "order_no"
                ),
                "status": ticket.get(
                    "status"
                ),
                "handler": ticket.get(
                    "handler"
                ),
                "created_at": ticket.get(
                    "created_at"
                ),
                "completed_at": ticket.get(
                    "completed_at"
                ),
                "detail": parse_json_field(
                    ticket.get(
                        "detail_json"
                    ),
                    {},
                ),
            }
        )

    for event in case.get(
        "timeline",
        [],
    ):
        result["timeline"].append(
            {
                "event_time": event.get(
                    "event_time"
                ),
                "event_type": event.get(
                    "event_type"
                ),
                "title": event.get(
                    "title"
                ),
                "status": event.get(
                    "status"
                ),
                "source_type": event.get(
                    "source_type"
                ),
                "source_id": event.get(
                    "source_id"
                ),
            }
        )

    return result


def get_retrieval_score(
    hit: dict[str, Any],
) -> float | None:
    """
    获取检索结果分数。
    混合检索结果使用 rrf_score；
    单独向量检索结果使用 score；
    单独 BM25 结果使用 bm25_score。
    """

    for field_name in (
        "rrf_score",
        "score",
        "bm25_score",
    ):
        value = hit.get(field_name)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    return None

def build_business_context(
    db_path: str | Path,
    current_session_id: str,
    history_hits: list[dict[str, Any]],
    max_history_cases: int = 3,
    as_of_message_id: str | None = None,
    as_of_message_seq: int | None = None,
) -> dict[str, Any]:
    """
    构建最终业务上下文。
    参数：
        db_path：SQLite 数据库路径。
        current_session_id：当前客服正在处理的会话 ID。
        history_hits：BM25、向量或混合检索返回的历史 Chunk。
        max_history_cases：最多补充多少个历史服务会话。
        as_of_message_id：只允许读取该消息及它之前的数据。
        as_of_message_seq：和 message_id 二选一，用消息序号设置截断点。
    """

    # 查询当前正在处理的会话。
    current_case_raw = get_case(
        db_path=db_path,
        session_id=current_session_id,
    )
    analysis_cutoff = resolve_analysis_cutoff(
        case=current_case_raw,
        as_of_message_id=as_of_message_id,
        as_of_message_seq=as_of_message_seq,
    )
    current_case_raw = filter_case_as_of(
        case=current_case_raw,
        cutoff=analysis_cutoff,
    )
    current_case = compact_case(
        current_case_raw,
        include_messages=True,
    )
    historical_evidence: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()

    for hit in history_hits:
        session_id = hit.get(
            "session_id"
        )
        if not session_id:
            continue
        session_id = str(session_id)

        # 当前会话不应该作为历史会话重复加入。
        if session_id == current_session_id:
            continue
        # 同一个会话只保留一次。
        if session_id in seen_session_ids:
            continue
        if len(historical_evidence) >= max_history_cases:
            break

        # 严格回放时，只允许使用在截断时间之前已经结束的历史片段。
        # 否则 RAG 虽然检索的是“历史库”，仍可能命中当天稍后才发生的会话。
        if analysis_cutoff is not None:
            cutoff_time = parse_datetime(
                analysis_cutoff.get("sent_at")
            )
            history_end_time = parse_datetime(
                hit.get("end_at")
            )
            if (
                cutoff_time is not None
                and (
                    history_end_time is None
                    or history_end_time > cutoff_time
                )
            ):
                continue

        try:
            historical_case_raw = get_case(
                db_path=db_path,
                session_id=session_id,
            )
        except KeyError:
            continue

        seen_session_ids.add(session_id)

        historical_evidence.append(
            {
                "retrieval": {
                    "chunk_id": hit.get(
                        "chunk_id"
                    ),
                    "parent_id": hit.get(
                        "parent_id"
                    ),
                    "session_id": session_id,
                    "score": get_retrieval_score(
                        hit
                    ),
                    "sources": hit.get(
                        "retrieval_sources",
                        [],
                    ),
                },

                # 这是向量库或 BM25 命中的历史文本。
                "matched_history_text": hit.get(
                    "text",
                    "",
                ),

                
                "business_facts": compact_case(
                    historical_case_raw,
                    include_messages=False,
                ),
            }
        )

    return {
        "analysis_cutoff": analysis_cutoff,
        "current_case": current_case,
        "historical_evidence": historical_evidence,
        "context_metadata": {
            "future_message_filter_applied": (
                analysis_cutoff is not None
            ),
            "historical_time_filter_applied": (
                analysis_cutoff is not None
            ),
            # 订单/工单表保存的是当前快照；若未来需要逐时刻精确恢复状态，
            # 还要建立状态变更事件表，而不能只依赖最终状态字段。
            "snapshot_status_may_be_current_state": True,
        },
    }

def render_context_text(
    context: dict[str, Any],
) -> str:
    """
    将结构化业务上下文转换成文本。
    这个文本后续可以直接放入 LLM Prompt。
    """
    lines: list[str] = []
    lines.append("【当前服务会话】")
    current_case = context["current_case"]
    current_session = current_case["session"]
    lines.append(
        f"会话 ID："
        f"{current_session.get('session_id')}"
    )
    lines.append(
        f"用户："
        f"{current_session.get('buyer_nickname')}"
    )
    lines.append(
        f"店铺："
        f"{current_session.get('store')}"
    )

    lines.append("")
    lines.append("【当前聊天记录】")

    for message in current_case.get(
        "messages",
        [],
    ):
        lines.append(
            f"{message.get('role')}："
            f"{message.get('message_text')}"
        )

    lines.append("")
    lines.append("【当前订单】")

    for order in current_case.get(
        "orders",
        [],
    ):
        lines.append(
            f"订单号：{order.get('order_no')}，"
            f"商品：{order.get('product_name')}，"
            f"状态：{order.get('order_status')}，"
            f"物流单号：{order.get('tracking_no')}"
        )

    lines.append("")
    lines.append("【当前工单】")

    for ticket in current_case.get(
        "tickets",
        [],
    ):
        lines.append(
            f"工单号：{ticket.get('ticket_no')}，"
            f"类型：{ticket.get('ticket_type')}，"
            f"状态：{ticket.get('status')}，"
            f"详情：{ticket.get('detail')}"
        )

    lines.append("")
    lines.append("【历史服务证据】")

    historical_evidence = context.get(
        "historical_evidence",
        [],
    )

    if not historical_evidence:
        lines.append("没有检索到相关历史服务记录")

    for index, evidence in enumerate(
        historical_evidence,
        start=1,
    ):
        retrieval = evidence["retrieval"]
        facts = evidence["business_facts"]
        session = facts["session"]

        lines.append(
            f"历史记录 {index}："
            f"会话 {retrieval.get('session_id')}，"
            f"检索来源："
            f"{retrieval.get('sources')}，"
            f"分数："
            f"{retrieval.get('score')}"
        )

        lines.append(
            f"历史聊天片段："
            f"{evidence.get('matched_history_text')}"
        )

        lines.append(
            f"历史用户："
            f"{session.get('buyer_nickname')}"
        )

        for order in facts.get(
            "orders",
            [],
        ):
            lines.append(
                f"历史订单："
                f"{order.get('order_no')}，"
                f"商品："
                f"{order.get('product_name')}，"
                f"状态："
                f"{order.get('order_status')}"
            )

        for ticket in facts.get(
            "tickets",
            [],
        ):
            lines.append(
                f"历史工单："
                f"{ticket.get('ticket_no')}，"
                f"类型："
                f"{ticket.get('ticket_type')}，"
                f"状态："
                f"{ticket.get('status')}"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build business context from SQLite and retrieval hits."
    )

    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="当前正在处理的会话 ID",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite 数据库路径",
    )

    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="混合检索结果 JSON 文件",
    )

    parser.add_argument(
        "--max-history-cases",
        type=int,
        default=3,
        help="最多补充多少个历史会话",
    )

    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--as-of-message-id",
        type=str,
        default=None,
        help="离线回放截断消息 ID，只读取该消息及之前的数据",
    )
    cutoff_group.add_argument(
        "--as-of-message-seq",
        type=int,
        default=None,
        help="离线回放截断消息序号，只读取该消息及之前的数据",
    )

    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="输出格式",
    )

    args = parser.parse_args()

    history_hits = load_history_hits(
        args.history_file
    )

    context = build_business_context(
        db_path=args.db,
        current_session_id=args.session_id,
        history_hits=history_hits,
        max_history_cases=args.max_history_cases,
        as_of_message_id=args.as_of_message_id,
        as_of_message_seq=args.as_of_message_seq,
    )

    if args.format == "json":
        print(
            json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print(
            render_context_text(context)
        )


if __name__ == "__main__":
    main()
