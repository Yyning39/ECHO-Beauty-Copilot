"""
    创建服务会话和历史记忆chunk
    输入：chat_messages.jsonl,orders.jsonl,ticket.jsonl
    输出：service_episodes.jsonl,memory_chunks.jsonl,memory_build_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def read_jsonl(path:Path)->list[dict[str,Any]]:
    records:list[dict[str,Any]]=[]
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with path.open("r",encoding="utf-8") as file:
        for line_number,line in enumerate(file,start=1):
            line=line.strip()
            if not line:
                continue
            try:
                record=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}第{line_number}行不是合法的json:{line}"
                )from exc
            if not isinstance(record,dict):
                raise ValueError(
                    f"{path}第{line_number}行不是字典对象"
                )
            records.append(record)
    return records

def write_jsonl(
    path:Path,
    records:list[dict[str,Any]],
)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +"\n"
            )
#从多条记录中提取某个字段的非空值，并去重
def get_non_empty_values(
    records:list[dict[str,Any]],
    field_name:str,
)->list[str]:
    values={
        str(record[field_name]).strip()
        for record in records
        if record.get(field_name)
    }
    return sorted(values)

def message_sort_key(
    message:dict[str,Any]
)->tuple[Any,str]:
    message_seq=message.get("message_seq")
    if isinstance(message_seq,int):
        sequence=message_seq
    elif isinstance(message_seq,float):
        sequence=int(message_seq)
    else:
        sequence=999999
    sent_at=str(message.get("sent_at") or "")
    return sequence,sent_at

def format_message(message:dict[str,Any])->str | None:
    message_text=str(message.get("message_text") or "").strip()
    if not message_text:
        return None
    role=str(message.get("role") or "").strip()
    if not role:
        if message.get("is_buyer_message") is True:
            role="买家"
        else:
            role="客服"
    return f"{role}:{message_text}"
#按sess_id聚合服务对话，service_episodes为一次完整的客户服务经历
def build_service_episodes(
    chats:list[dict[str,Any]],
    orders:list[dict[str,Any]],
    tickets:list[dict[str,Any]],
)->list[dict[str,Any]]:
    chats_by_session:dict[str,list[dict[str,Any]]]=defaultdict(list)
    orders_by_session:dict[str,list[dict[str,Any]]]=defaultdict(list)
    tickets_by_session:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for chat in chats:
        session_id=chat.get("session_id")
        if session_id:
            chats_by_session[str(session_id)].append(chat)
    for order in orders:
        session_id=order.get("session_id")
        if session_id:
            orders_by_session[str(session_id)].append(order)
    for ticket in tickets:
        session_id=ticket.get("session_id")
        if session_id:
            tickets_by_session[str(session_id)].append(ticket)
    all_session_ids=(
        set(chats_by_session)
        | set(orders_by_session)
        | set(tickets_by_session)
    )
    episodes:list[dict[str,Any]]=[]
    for session_id in sorted(all_session_ids):
        session_chats=sorted(
            chats_by_session.get(session_id,[]),
            key=message_sort_key,
        )
        session_orders=orders_by_session.get(session_id,[])
        session_tickets=tickets_by_session.get(session_id,[])
        buyer_names=get_non_empty_values(
            session_chats+session_orders+session_tickets,
            "buyer_nickname",
        )
        chat_times=[
            str(chat["sent_at"])
            for chat in session_chats
            if chat.get("sent_at")
        ]
        formatted_messages:list[dict[str,Any]]=[]
        for chat in session_chats:
            formatted_text=format_message(chat)
            if formatted_text is None:
                continue
            formatted_messages.append(
                {"message_id":chat.get("message_id"),
                "message_seq":chat.get("message_seq"),
                "message_at":chat.get("message_at"),
                "role":chat.get("role"),
                "text":formatted_text,}
            )
        #找到用户在本次对话中的第一句话
        first_buyer_message=None
        for chat in session_chats:
            if chat.get("is_buyer_message") is True:
                text=str(chat.get("message_text") or "").strip()
                if text:
                    first_buyer_message=text
                    break
        episode={
            # 父级记录的唯一 ID。
            "episode_id": f"session:{session_id}",
            # 这次服务会话的 ID。
            "session_id": session_id,
            # 当前数据中使用买家昵称作为用户标识。
            # 真实项目中应该替换为 buyer_id。
            "buyer_nicknames": buyer_names,
            # 一次会话可能关联订单和工单。
            "order_nos": sorted(
                {
                    str(order["order_no"])
                    for order in session_orders
                    if order.get("order_no")
                }
            ),
            "ticket_nos": sorted(
                {
                    str(ticket["ticket_no"])
                    for ticket in session_tickets
                    if ticket.get("ticket_no")
                }
            ),

                # 会话开始和结束时间。
                "start_at": min(chat_times) if chat_times else None,
                "end_at": max(chat_times) if chat_times else None,

                # 便于前端展示和后续检索。
                "first_buyer_message": first_buyer_message,
                "message_count": len(formatted_messages),
                "order_count": len(session_orders),
                "ticket_count": len(session_tickets),

                # 保存会话中的标准化聊天消息。
                "messages": formatted_messages,

                # 保存订单和工单的简要结构。
                # 完整订单和工单事实仍然建议从 SQLite 查询。
                "orders": session_orders,
                "tickets": session_tickets,
            }
        episodes.append(episode)
    return episodes

def build_memory_chunks(
    episodes:list[dict[str,Any]],
    max_chars:int=800,
    overlap_messages:int=1,
)->list[dict[str,Any]]:
    chunks:list[dict[str,Any]]=[]
    for episode in episodes:
        messages=episode.get("messages") or []
        if not messages:
            continue
        current_messages:list[dict[str,Any]]=[]
        current_length=0
        chunk_index=0

        def emit_chunk(
            chunk_messages:list[dict[str,Any]],
            index:int,
        )->dict[str,Any]:
            session_id=episode["session_id"]
            text_lines=[
                f"历史服务会话：{session_id}"
            ]
            for message in chunk_messages:
                text_lines.append(str(message["text"]))
            chunk_text="\n".join(text_lines)
            times=[
                str(message["sent_at"])
                for message in chunk_messages
                if message.get("sent_at")
            ]

            return {
                # 子 Chunk 的唯一 ID。
                "chunk_id": f"{session_id}:chunk:{index:03d}",

                # 指向完整服务会话。
                "parent_id": episode["episode_id"],

                "session_id": session_id,
                "buyer_nicknames": episode["buyer_nicknames"],
                "order_nos": episode["order_nos"],
                "ticket_nos": episode["ticket_nos"],

                "chunk_index": index,
                "message_count": len(chunk_messages),
                "start_at": min(times) if times else None,
                "end_at": max(times) if times else None,

                # 未来会对这个字段做 Embedding。
                "text": chunk_text,

                # 保存消息 ID，方便追溯来源。
                "message_ids": [
                    message.get("message_id")
                    for message in chunk_messages
                    if message.get("message_id")
                ],
            }
        for message in messages:
            message_text = str(message.get("text") or "")
            message_length = len(message_text)

            # 如果当前 Chunk 已经有内容，并且继续加入会超过长度，
            # 就先输出当前 Chunk。
            if (
                current_messages
                and current_length + message_length > max_chars
            ):
                chunks.append(
                    emit_chunk(
                        current_messages,
                        chunk_index,
                    )
                )

                chunk_index += 1

                # 保留最后几条消息作为 overlap。
                # 这样下一个 Chunk 可以保留一部分上下文。
                if overlap_messages > 0:
                    current_messages = current_messages[
                        -overlap_messages:
                    ]
                else:
                    current_messages = []

                current_length = sum(
                    len(str(item.get("text") or ""))
                    for item in current_messages
                )

            current_messages.append(message)
            current_length += message_length

        # 循环结束后，不要忘记输出最后一个 Chunk。
        if current_messages:
            chunks.append(
                emit_chunk(
                    current_messages,
                    chunk_index,
                )
            )

    return chunks

def build_report(episodes: list[dict[str, Any]],chunks: list[dict[str, Any]],) -> dict[str, Any]:
    episode_ids = {
        episode["episode_id"]
        for episode in episodes
    }
    missing_parent_ids = sorted(
        {
            chunk["parent_id"]
            for chunk in chunks
            if chunk.get("parent_id") not in episode_ids
        }
    )

    return {
        "episode_count": len(episodes),
        "chunk_count": len(chunks),
        "episodes_with_messages": sum(
            1
            for episode in episodes
            if episode.get("message_count", 0) > 0
        ),
        "episodes_without_messages": sum(
            1
            for episode in episodes
            if episode.get("message_count", 0) == 0
        ),
        "chunks_without_parent": len(missing_parent_ids),
        "missing_parent_ids": missing_parent_ids,
        "ready_for_embedding": (
            len(chunks) > 0
            and len(missing_parent_ids) == 0
        ),
    } 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build service episodes and historical memory chunks."
    )

    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cleaned",
        help="清洗后 JSONL 文件所在目录",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory",
        help="会话和 Chunk 输出目录",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="单个 Chunk 最大字符数",
    )

    parser.add_argument(
        "--overlap-messages",
        type=int,
        default=1,
        help="相邻 Chunk 重叠的消息数量",
    )

    args = parser.parse_args()

    # 读取前一个阶段生成的清洗数据。
    chats = read_jsonl(
        args.cleaned_dir / "chat_messages.jsonl"
    )

    orders = read_jsonl(
        args.cleaned_dir / "orders.jsonl"
    )

    tickets = read_jsonl(
        args.cleaned_dir / "tickets.jsonl"
    )

    # 第一步：按会话聚合聊天、订单和工单。
    episodes = build_service_episodes(
        chats=chats,
        orders=orders,
        tickets=tickets,
    )

    # 第二步：将会话中的连续聊天消息切成子 Chunk。
    chunks = build_memory_chunks(
        episodes=episodes,
        max_chars=args.max_chars,
        overlap_messages=args.overlap_messages,
    )

    # 第三步：检查 Chunk 是否能够找到对应的父会话。
    report = build_report(
        episodes=episodes,
        chunks=chunks,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 输出完整服务会话。
    write_jsonl(
        args.output_dir / "service_episodes.jsonl",
        episodes,
    )

    # 输出用于未来 Embedding 的子 Chunk。
    write_jsonl(
        args.output_dir / "memory_chunks.jsonl",
        chunks,
    )

    # 输出构建报告。
    (args.output_dir / "memory_build_report.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()