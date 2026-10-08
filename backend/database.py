"""SQLite schema and competition workbook importer for ECHO Beauty."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "echo_beauty.sqlite3"

CHAT_SHEET = "聊天记录"
ORDER_SHEET = "订单"
TICKET_SHEETS = {
    "补发换货工单": "补发换货",
    "线下打款工单": "线下打款",
    "物流工单": "物流",
    "不良反应工单": "不良反应",
    "售后退货工单": "售后退货",
}


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS import_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    row_counts_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS buyers (
    buyer_nickname TEXT PRIMARY KEY,
    first_seen_at TEXT,
    last_seen_at TEXT,
    session_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    buyer_nickname TEXT,
    store TEXT,
    first_message_at TEXT,
    last_message_at TEXT,
    FOREIGN KEY (buyer_nickname) REFERENCES buyers(buyer_nickname)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_seq INTEGER,
    message_id TEXT,
    sent_at TEXT,
    role TEXT,
    buyer_nickname TEXT,
    sender TEXT,
    store TEXT,
    scene_major TEXT,
    scene_minor TEXT,
    is_target_buyer_message INTEGER,
    message_text TEXT,
    content_type TEXT,
    chat_content TEXT,
    category TEXT,
    image_path TEXT,
    related_order_no TEXT,
    related_ticket_no TEXT,
    raw_json TEXT NOT NULL,
    UNIQUE(session_id, message_seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_no TEXT PRIMARY KEY,
    session_id TEXT,
    buyer_nickname TEXT,
    store TEXT,
    product_sku TEXT,
    product_name TEXT,
    quantity REAL,
    unit_price REAL,
    paid_amount REAL,
    order_status TEXT,
    ordered_at TEXT,
    paid_at TEXT,
    shipped_at TEXT,
    carrier TEXT,
    tracking_no TEXT,
    province TEXT,
    city TEXT,
    gift TEXT,
    buyer_note TEXT,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (buyer_nickname) REFERENCES buyers(buyer_nickname)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_no TEXT PRIMARY KEY,
    ticket_type TEXT NOT NULL,
    source_sheet TEXT NOT NULL,
    session_id TEXT,
    order_no TEXT,
    buyer_nickname TEXT,
    store TEXT,
    status TEXT,
    handler TEXT,
    created_at TEXT,
    completed_at TEXT,
    detail_json TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (order_no) REFERENCES orders(order_no),
    FOREIGN KEY (buyer_nickname) REFERENCES buyers(buyer_nickname)
);

CREATE TABLE IF NOT EXISTS service_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    buyer_nickname TEXT,
    event_time TEXT,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    UNIQUE(source_type, source_id, event_type, event_time),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (buyer_nickname) REFERENCES buyers(buyer_nickname)
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    buyer_nickname TEXT,
    risk_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '待处理',
    owner TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (buyer_nickname) REFERENCES buyers(buyer_nickname)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    buyer_nickname TEXT,
    trace_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,
    intent_json TEXT,
    retrieval_json TEXT,
    reply_json TEXT,
    token_usage_json TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS reply_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,
    original_reply TEXT NOT NULL,
    final_reply TEXT,
    reason TEXT,
    analysis_schema_version TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_session_time
    ON chat_messages(session_id, sent_at);
CREATE INDEX IF NOT EXISTS idx_chat_buyer_time
    ON chat_messages(buyer_nickname, sent_at);
CREATE INDEX IF NOT EXISTS idx_orders_session
    ON orders(session_id);
CREATE INDEX IF NOT EXISTS idx_orders_buyer
    ON orders(buyer_nickname);
CREATE INDEX IF NOT EXISTS idx_tickets_session
    ON tickets(session_id);
CREATE INDEX IF NOT EXISTS idx_tickets_order
    ON tickets(order_no);
CREATE INDEX IF NOT EXISTS idx_timeline_session_time
    ON service_events(session_id, event_time);
CREATE INDEX IF NOT EXISTS idx_reply_feedback_session_time
    ON reply_feedback(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_risk_status
    ON risk_events(status, risk_level);
"""


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    return int(str(value).strip().lower() in {"1", "true", "yes", "是", "y"})


def _json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, default=str)


def _read_sheet(workbook: Any, sheet_name: str) -> list[dict[str, Any]]:
    sheet = workbook[sheet_name]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    result: list[dict[str, Any]] = []
    for values in rows:
        if not any(value is not None and str(value).strip() for value in values):
            continue
        result.append({header: value for header, value in zip(headers, values) if header})
    return result


def create_database(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _upsert_buyers_and_sessions(
    connection: sqlite3.Connection,
    chats: Iterable[dict[str, Any]],
    orders: Iterable[dict[str, Any]],
    tickets: Iterable[dict[str, Any]],
) -> None:
    buyer_values: dict[str, dict[str, Any]] = {}
    session_values: dict[str, dict[str, Any]] = {}

    for row in [*chats, *orders, *tickets]:
        buyer = _text(row.get("买家昵称"))
        session = _text(row.get("会话ID"))
        store = _text(row.get("店铺"))
        if buyer:
            buyer_values.setdefault(buyer, {"buyer_nickname": buyer})
        if session:
            session_values.setdefault(
                session,
                {"session_id": session, "buyer_nickname": buyer, "store": store},
            )
            if buyer:
                session_values[session]["buyer_nickname"] = buyer
            if store:
                session_values[session]["store"] = store

    connection.executemany(
        "INSERT OR IGNORE INTO buyers(buyer_nickname) VALUES (?)",
        [(key,) for key in buyer_values],
    )
    connection.executemany(
        """
        INSERT INTO sessions(session_id, buyer_nickname, store)
        VALUES (:session_id, :buyer_nickname, :store)
        ON CONFLICT(session_id) DO UPDATE SET
          buyer_nickname=COALESCE(excluded.buyer_nickname, sessions.buyer_nickname),
          store=COALESCE(excluded.store, sessions.store)
        """,
        list(session_values.values()),
    )


def _insert_chats(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    sql = """
    INSERT OR REPLACE INTO chat_messages(
      session_id, message_seq, message_id, sent_at, role, buyer_nickname,
      sender, store, scene_major, scene_minor, is_target_buyer_message,
      message_text, content_type, chat_content, category, image_path,
      related_order_no, related_ticket_no, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = []
    for row in rows:
        session = _text(row.get("会话ID"))
        if not session:
            continue
        values.append(
            (
                session,
                int(row["消息序号"]) if row.get("消息序号") is not None else None,
                _text(row.get("message_id")),
                _text(row.get("发送时间")),
                _text(row.get("角色")),
                _text(row.get("买家昵称")),
                _text(row.get("发送方")),
                _text(row.get("店铺")),
                _text(row.get("scene_major")),
                _text(row.get("scene_minor")),
                _flag(row.get("is_target_buyer_message")),
                _text(row.get("message_text")),
                _text(row.get("内容类型")),
                _text(row.get("chat_content")),
                _text(row.get("category")),
                _text(row.get("image_path")),
                _text(row.get("关联订单号")),
                _text(row.get("关联工单号")),
                _json(row),
            )
        )
    connection.executemany(sql, values)


def _insert_orders(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    sql = """
    INSERT OR REPLACE INTO orders(
      order_no, session_id, buyer_nickname, store, product_sku, product_name,
      quantity, unit_price, paid_amount, order_status, ordered_at, paid_at,
      shipped_at, carrier, tracking_no, province, city, gift, buyer_note, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = []
    for row in rows:
        order_no = _text(row.get("订单号"))
        if not order_no:
            continue
        values.append(
            (
                order_no,
                _text(row.get("会话ID")),
                _text(row.get("买家昵称")),
                _text(row.get("店铺")),
                _text(row.get("商品货号")),
                _text(row.get("商品名称")),
                _number(row.get("数量")),
                _number(row.get("单价(元)")),
                _number(row.get("实付金额(元)")),
                _text(row.get("订单状态")),
                _text(row.get("下单时间")),
                _text(row.get("付款时间")),
                _text(row.get("发货时间")),
                _text(row.get("快递公司")),
                _text(row.get("物流单号")),
                _text(row.get("收货省")),
                _text(row.get("收货市")),
                _text(row.get("赠品")),
                _text(row.get("买家留言")),
                _json(row),
            )
        )
    connection.executemany(sql, values)


def _insert_tickets(
    connection: sqlite3.Connection,
    ticket_rows: list[tuple[str, str, dict[str, Any]]],
) -> None:
    sql = """
    INSERT OR REPLACE INTO tickets(
      ticket_no, ticket_type, source_sheet, session_id, order_no,
      buyer_nickname, store, status, handler, created_at, completed_at,
      detail_json, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = []
    for source_sheet, ticket_type, row in ticket_rows:
        ticket_no = _text(row.get("工单号"))
        if not ticket_no:
            continue
        status = _text(row.get("工单状态")) or _text(row.get("任务状态")) or _text(row.get("转账状态"))
        excluded = {"工单号", "会话ID", "关联订单号", "买家昵称", "店铺", "工单状态", "任务状态", "转账状态", "处理人", "创建时间", "完成时间"}
        detail = {key: value for key, value in row.items() if key not in excluded}
        values.append(
            (
                ticket_no,
                ticket_type,
                source_sheet,
                _text(row.get("会话ID")),
                _text(row.get("关联订单号")),
                _text(row.get("买家昵称")),
                _text(row.get("店铺")),
                status,
                _text(row.get("处理人")),
                _text(row.get("创建时间")),
                _text(row.get("完成时间")),
                _json(detail),
                _json(row),
            )
        )
    connection.executemany(sql, values)


def _insert_service_events(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM service_events")

    connection.execute(
        """
        INSERT OR IGNORE INTO service_events(
          session_id, buyer_nickname, event_time, event_type, title, status,
          source_type, source_id, detail_json
        )
        SELECT session_id, buyer_nickname, sent_at, 'chat',
               CASE WHEN role = '买家' THEN '买家消息' ELSE '客服消息' END,
               NULL, 'chat', COALESCE(message_id, CAST(id AS TEXT)), raw_json
        FROM chat_messages
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO service_events(
          session_id, buyer_nickname, event_time, event_type, title, status,
          source_type, source_id, detail_json
        )
        SELECT session_id, buyer_nickname, ordered_at, 'order', '创建订单', order_status,
               'order', order_no, raw_json
        FROM orders
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO service_events(
          session_id, buyer_nickname, event_time, event_type, title, status,
          source_type, source_id, detail_json
        )
        SELECT session_id, buyer_nickname, created_at, 'ticket', ticket_type || '工单', status,
               'ticket', ticket_no, raw_json
        FROM tickets
        """
    )

    connection.execute(
        """
        INSERT INTO buyers(buyer_nickname, first_seen_at, last_seen_at, session_count)
        SELECT buyer_nickname, MIN(event_time), MAX(event_time), COUNT(DISTINCT session_id)
        FROM service_events
        WHERE buyer_nickname IS NOT NULL
        GROUP BY buyer_nickname
        ON CONFLICT(buyer_nickname) DO UPDATE SET
          first_seen_at=excluded.first_seen_at,
          last_seen_at=excluded.last_seen_at,
          session_count=excluded.session_count
        """
    )
    connection.execute(
        """
        UPDATE sessions
        SET first_message_at=(SELECT MIN(event_time) FROM service_events WHERE service_events.session_id = sessions.session_id),
            last_message_at=(SELECT MAX(event_time) FROM service_events WHERE service_events.session_id = sessions.session_id)
        """
    )


def import_workbook(
    workbook_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Import the supplied competition workbook into SQLite and rebuild timeline events."""

    from openpyxl import load_workbook

    source = Path(workbook_path)
    if not source.exists():
        raise FileNotFoundError(f"Workbook not found: {source}")

    workbook = load_workbook(source, read_only=True, data_only=True)
    chats = _read_sheet(workbook, CHAT_SHEET)
    orders = _read_sheet(workbook, ORDER_SHEET)
    ticket_rows: list[tuple[str, str, dict[str, Any]]] = []
    for sheet_name, ticket_type in TICKET_SHEETS.items():
        ticket_rows.extend((sheet_name, ticket_type, row) for row in _read_sheet(workbook, sheet_name))
    tickets = [row for _, _, row in ticket_rows]

    connection = create_database(db_path)
    try:
        _upsert_buyers_and_sessions(connection, chats, orders, tickets)
        _insert_chats(connection, chats)
        _insert_orders(connection, orders)
        _insert_tickets(connection, ticket_rows)
        _insert_service_events(connection)
        counts = {
            "buyers": connection.execute("SELECT COUNT(*) FROM buyers").fetchone()[0],
            "sessions": connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "chat_messages": connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0],
            "orders": connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "tickets": connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
            "service_events": connection.execute("SELECT COUNT(*) FROM service_events").fetchone()[0],
        }
        connection.execute(
            "INSERT INTO import_runs(source_file, imported_at, row_counts_json) VALUES (?, ?, ?)",
            (str(source), datetime.now().isoformat(timespec="seconds"), _json(counts)),
        )
        connection.commit()
        return counts
    finally:
        connection.close()


def get_case(db_path: str | Path, session_id: str) -> dict[str, Any]:
    """Return the fact-based case payload used by the future Agent API."""

    connection = create_database(db_path)
    try:
        session = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise KeyError(f"Unknown session_id: {session_id}")
        chats = connection.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY sent_at, message_seq",
            (session_id,),
        ).fetchall()
        orders = connection.execute(
            "SELECT * FROM orders WHERE session_id = ? ORDER BY ordered_at", (session_id,)
        ).fetchall()
        tickets = connection.execute(
            "SELECT * FROM tickets WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        timeline = connection.execute(
            "SELECT * FROM service_events WHERE session_id = ? ORDER BY event_time, id",
            (session_id,),
        ).fetchall()

        def rows(items: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
            return [dict(item) for item in items]

        return {
            "session": dict(session),
            "messages": rows(chats),
            "orders": rows(orders),
            "tickets": rows(tickets),
            "timeline": rows(timeline),
        }
    finally:
        connection.close()


def list_session_ids(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 20) -> list[str]:
    """List recently active sessions for the workbench inbox."""

    connection = create_database(db_path)
    try:
        rows = connection.execute(
            """
            SELECT session_id
            FROM sessions
            ORDER BY COALESCE(last_message_at, first_message_at) DESC, session_id
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        connection.close()


def create_risk_event(
    db_path: str | Path,
    *,
    session_id: str,
    buyer_nickname: str | None,
    risk_type: str,
    risk_level: str,
    reason: str,
    evidence: list[str],
    owner: str | None = None,
) -> int:
    """Persist a risk event created by a human or the future Agent."""

    now = datetime.now().isoformat(timespec="seconds")
    connection = create_database(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO risk_events(
              session_id, buyer_nickname, risk_type, risk_level, reason,
              evidence_json, status, owner, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, '待处理', ?, ?, ?)
            """,
            (
                session_id,
                buyer_nickname,
                risk_type,
                risk_level,
                reason,
                json.dumps(evidence, ensure_ascii=False),
                owner,
                now,
                now,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def update_risk_event(
    db_path: str | Path,
    *,
    event_id: int,
    status: str,
    owner: str | None = None,
) -> bool:
    """更新风险事件的处理状态，形成待处理→处理中→已解决闭环。"""

    allowed_statuses = {"待处理", "处理中", "已解决", "已关闭"}
    if status not in allowed_statuses:
        raise ValueError(f"不支持的风险状态：{status}")

    now = datetime.now().isoformat(timespec="seconds")
    connection = create_database(db_path)
    try:
        cursor = connection.execute(
            """
            UPDATE risk_events
            SET status = ?, owner = COALESCE(?, owner), updated_at = ?
            WHERE id = ?
            """,
            (status, owner, now, event_id),
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def create_reply_feedback(
    db_path: str | Path,
    *,
    session_id: str,
    action: str,
    original_reply: str,
    final_reply: str | None = None,
    reason: str | None = None,
    analysis_schema_version: str | None = None,
) -> int:
    """保存人工客服对 AI 回复的采纳、编辑或拒绝结果。

    这些反馈以后可以用于评估“建议采纳率”和定位需要优化的场景，而不是
    直接拿来自动训练模型。
    """

    allowed_actions = {"accepted", "edited", "rejected"}
    if action not in allowed_actions:
        raise ValueError(f"不支持的回复反馈动作：{action}")
    if not original_reply.strip():
        raise ValueError("original_reply 不能为空")

    now = datetime.now().isoformat(timespec="seconds")
    connection = create_database(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO reply_feedback(
              session_id, action, original_reply, final_reply,
              reason, analysis_schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                action,
                original_reply,
                final_reply,
                reason,
                analysis_schema_version,
                now,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def create_agent_run(
    db_path: str | Path,
    *,
    session_id: str,
    buyer_nickname: str | None,
    trace_id: str,
    status: str,
    intent: dict[str, Any],
    retrieval: dict[str, Any],
    reply: dict[str, Any],
    token_usage: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> int:
    """保存一次完整 Agent 调用摘要，供耗时、效果和失败率统计。"""

    now = datetime.now().isoformat(timespec="seconds")
    connection = create_database(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO agent_runs(
              session_id, buyer_nickname, trace_id, started_at, finished_at,
              status, intent_json, retrieval_json, reply_json,
              token_usage_json, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                buyer_nickname,
                trace_id,
                now,
                now,
                status,
                json.dumps(intent, ensure_ascii=False),
                json.dumps(retrieval, ensure_ascii=False),
                json.dumps(reply, ensure_ascii=False),
                json.dumps(token_usage or {}, ensure_ascii=False),
                error_message,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()
