#数据清洗
from __future__ import annotations
import json
import re
from collections import Counter,defaultdict
from datetime import date,datetime
from pathlib import Path
from typing import Any,Iterable

#Excle 配置

#聊天记录所在的sheet名称
CHAT_SHEET="聊天记录"
ORDER_SHEET="订单"
TICKET_SHEETS = {
    "补发换货工单": "补发换货",
    "线下打款工单": "线下打款",
    "物流工单": "物流工单",
    "不良反应工单": "不良反应",
    "售后退货工单": "售后退货",
}
#基础字段清洗函数
EMPTY_VALUES = {
    "",
    "-",
    "--",
    "无",
    "暂无",
    "none",
    "null",
    "nan",
}
def clean_text(value:Any)->str|None:
    if value is None:
        return None
    #统一转化为字符串
    if isinstance(value,datetime):
        return value.isoformat(sep="T")
    if isinstance(value,date):
        return value.isoformat()
    
    text=str(value).replace("\u00a0"," ")
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    text=text.strip()
    if text.lower() in EMPTY_VALUES:
        return None
    return text

def clean_datetime(value:Any)->str | None:
    if value is None or value=="":
        return None
    if isinstance(value,datetime):
        return value.isoformat(sep="T")
    if isinstance(value,date):
        return value.isoformat()
    text=clean_text(value)
    if not text:
        return None
    # 尝试解析几种常见时间格式。
    supported_formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in supported_formats:
        try:
            parsed=datetime.strptime(text,fmt)
            return parsed.isoformat(sep="T")
        except ValueError:
            continue
    return text

def clean_number(value:Any)->float | None:
    if value is None or value=="":
        return None
    if isinstance(value,(int,float)):
        return float(value)
    text=clean_text(value)
    if not text:
        return None
    text=text.replace(",","")
    text=text.replace("¥","")
    text=text.replace("元","")
    try:
        return float(text)
    except ValueError:
        return None

def clean_integer(value:Any)->int | None:
    number=clean_number(value)
    if number is None:
        return None
    return int(number)
def clean_bool(value:Any)->bool | None:
    if value is None or value=="":
        return None
    if isinstance(value,bool):
        return value
    text=str(value).strip().lower()
    if text in {"1","true","yes","y","是"}:
        return True
    if text in {"0","false","no","n","否"}:
        return False
    return None
def normalize_status(value:Any)->str | None:
    text=clean_text(value)
    if not text:
        return None
    status_mapping = {
        "已完成": "完成",
        "处理完成": "完成",
        "完成": "完成",
        "已关闭": "关闭",
        "关闭": "关闭",
        "处理中": "处理中",
        "工单组处理": "处理中",
        "仓库处理中": "处理中",
        "待处理": "待处理",
        "待审核": "待处理",
    }
    return status_mapping.get(text,text)

#excel读取
def read_sheet(workbook:Any,sheet_name:str)->list[dict[str,Any]]:
    sheet=workbook[sheet_name]
    #values_only=True只读取单元的值
    rows=sheet.iter_rows(values_only=True)
    headers=[]
    for value in next(rows):
        headers.append(clean_text(value) or "")
    result=[]
    for values in rows:
        if not any(
            value is not None and str(value).strip()
            for value in values
        ):
            continue
        row={
            header:value
            for header,value in zip(headers,values)
            if header
        }
        result.append(row)
    return result

#三类业务数据清洗
def clean_chat(row:dict[str,Any])->dict[str,Any] | None:
    session_id=clean_text(row.get("会话ID"))
    if not session_id:
        return None
    role=clean_text(row.get("角色"))
    is_buyer=clean_bool(
        row.get("is_target_buyer_message")
    )
    if is_buyer is None and role:
        is_buyer=(
            "买家" in role or "消费者" in role
        )
    message_text=(
        clean_text(row.get("message_text")) or clean_text(row.get("chat_content"))
    )
    return {
        "session_id": session_id,
        "message_seq": clean_integer(row.get("消息序号")),
        "message_id": clean_text(row.get("message_id")),
        "sent_at": clean_datetime(row.get("发送时间")),
        "role": role,
        "buyer_nickname": clean_text(row.get("买家昵称")),
        "sender": clean_text(row.get("发送方")),
        "store": clean_text(row.get("店铺")),
        "scene_major": clean_text(row.get("scene_major")),
        "scene_minor": clean_text(row.get("scene_minor")),
        "is_buyer_message": is_buyer,
        "message_text": message_text,
        "content_type": clean_text(row.get("内容类型")),
        "image_path": clean_text(row.get("image_path")),
        "related_order_no": clean_text(row.get("关联订单号")),
        "related_ticket_no": clean_text(row.get("关联工单号")),
        "raw":row,
    }
def clean_order(row:dict[str,Any])->dict[str,Any] | None:
    order_no=clean_text(row.get("订单号"))
    if not order_no:
        return None
    return {
        "order_no": order_no,
        "session_id": clean_text(row.get("会话ID")),
        "buyer_nickname": clean_text(row.get("买家昵称")),
        "store": clean_text(row.get("店铺")),
        "product_sku": clean_text(row.get("商品货号")),
        "product_name": clean_text(row.get("商品名称")),
        "quantity": clean_number(row.get("数量")),
        "unit_price": clean_number(row.get("单价(元)")),
        "paid_amount": clean_number(row.get("实付金额(元)")),
        "order_status": normalize_status(row.get("订单状态")),
        "ordered_at": clean_datetime(row.get("下单时间")),
        "paid_at": clean_datetime(row.get("付款时间")),
        "shipped_at": clean_datetime(row.get("发货时间")),
        "carrier": clean_text(row.get("快递公司")),
        "tracking_no": clean_text(row.get("物流单号")),
        "province": clean_text(row.get("收货省")),
        "city": clean_text(row.get("收货市")),
        "gift": clean_text(row.get("赠品")),
        "buyer_note": clean_text(row.get("买家留言")),
        "raw": row,
    }
def clean_ticket(
        row:dict[str,Any],
        source_sheet:str,
        ticket_type:str,
)->dict[str,Any] | None:
    ticket_no=clean_text(row.get("工单号"))
    if not ticket_no:
        return None
    status_value = (
        row.get("工单状态")
        or row.get("任务状态")
        or row.get("转账状态")
    )
    fixed_fields = {
        "工单号",
        "会话ID",
        "关联订单号",
        "买家昵称",
        "店铺",
        "工单状态",
        "任务状态",
        "转账状态",
        "处理人",
        "创建时间",
        "完成时间",
    }
    # 剩余字段都作为工单的业务详情。
    detail = {
        key: clean_text(value)
        for key, value in row.items()
        if key not in fixed_fields
    }
    return {
        "ticket_no": ticket_no,
        "ticket_type": ticket_type,
        "source_sheet": source_sheet,
        "session_id": clean_text(row.get("会话ID")),
        "order_no": clean_text(row.get("关联订单号")),
        "buyer_nickname": clean_text(row.get("买家昵称")),
        "store": clean_text(row.get("店铺")),
        "status_raw": clean_text(status_value),
        "status": normalize_status(status_value),
        "handler": clean_text(row.get("处理人")),
        "created_at": clean_datetime(row.get("创建时间")),
        "completed_at": clean_datetime(row.get("完成时间")),
        "detail": detail,
        "raw": row,
    }
def load_and_clean_workbook(
    workbook_path:str | Path,
)->dict[str,list[dict[str,Any]]]:
    from openpyxl import load_workbook
    workbook=load_workbook(
        workbook_path,
        read_only=True,
        data_only=True,
    )
    #清洗聊天记录
    chats=[]
    for row in read_sheet(workbook,CHAT_SHEET):
        cleaned=clean_chat(row)
        if cleaned is not None:
            chats.append(cleaned)
    #清洗订单
    orders=[]
    for row in read_sheet(workbook,ORDER_SHEET):
        cleaned=clean_order(row)
        if cleaned is not None:
            orders.append(cleaned)
    #读取并统一所有工单
    tickets=[]
    for sheet_name,ticket_type in TICKET_SHEETS.items():
        rows=read_sheet(workbook,sheet_name)
        for row in rows:
            cleaned=clean_ticket(
                row=row,
                source_sheet=sheet_name,
                ticket_type=ticket_type
            )
            if cleaned is not None:
                tickets.append(cleaned)
    return{
        "chats":chats,
        "orders":orders,
        "tickets":tickets,
    }
#数据质量质检
def find_duplicates(
    values:Iterable[str | None],
)->list[str]:
    counts=Counter(
        value
        for value in values
        if value
    )
    return sorted(
        value
        for value,count in counts.items()
        if count>1
    )
def sample_values(
    values:Iterable[str],
    limit:int=20
)->list[str]:
    """
    只保留问题样例，避免质量报告过大。
    """
    return sorted(set(values))[:limit]
def validate_relations(
    data:dict[str,list[dict[str,Any]]],
)->dict[str,Any]:
    chats = data["chats"]
    orders = data["orders"]
    tickets = data["tickets"]

    #建立订单号索引
    orders_by_no={
        row["order_no"]:row
        for row in orders
        if row.get("order_no")
    }
    #建立工单号索引
    tickets_by_no={
        row["ticket_no"]:row
        for row in tickets
        if row.get("ticket_no")
    }
    #建立session_id到订单号的映射
    orders_by_session:dict[str,set[str]]=defaultdict(set)
    for row in orders:
        session_id=row.get("session_id")
        order_no=row.get("order_no")
        if session_id and order_no:
            orders_by_session[session_id].add(order_no)
    # 建立 session_id 到工单号集合的映射。
    tickets_by_session: dict[str, set[str]] = defaultdict(set)
    for row in tickets:
        session_id = row.get("session_id")
        ticket_no = row.get("ticket_no")

        if session_id and ticket_no:
            tickets_by_session[session_id].add(ticket_no)
    # 检查聊天中明确填写的订单号是否存在。
    chat_order_unmatched = []
    for row in chats:
        order_no = row.get("related_order_no")

        if order_no and order_no not in orders_by_no:
            chat_order_unmatched.append(order_no)

    # 检查聊天中明确填写的工单号是否存在。
    chat_ticket_unmatched = []
    for row in chats:
        ticket_no = row.get("related_ticket_no")

        if ticket_no and ticket_no not in tickets_by_no:
            chat_ticket_unmatched.append(ticket_no)
    # 赛事数据说明中提到：
    # 一个会话至多一笔订单。
    #
    # 如果一个会话关联多个订单，
    # 后续查询时就不能直接默认取第一笔。
    sessions_with_multiple_orders = [
        session_id
        for session_id, order_nos in orders_by_session.items()
        if len(order_nos) > 1
    ]

    #同理检查一个会话是否关联多个工单。
    sessions_with_multiple_tickets = [
        session_id
        for session_id, ticket_nos in tickets_by_session.items()
        if len(ticket_nos) > 1
    ]

    #统计买家数量。
    buyers = {
        row.get("buyer_nickname")
        for row in chats + orders + tickets
        if row.get("buyer_nickname")
    }

    #统计会话数量。
    chat_sessions = {
        row.get("session_id")
        for row in chats
        if row.get("session_id")
    }

    all_sessions = (
        chat_sessions
        | set(orders_by_session.keys())
        | set(tickets_by_session.keys())
    )

    records = {
        "chats": len(chats),
        "orders": len(orders),
        "tickets": len(tickets),
        "buyers": len(buyers),
        "sessions": len(all_sessions),
    }

    duplicate_order_nos = find_duplicates(
        row.get("order_no")
        for row in orders
    )

    duplicate_ticket_nos = find_duplicates(
        row.get("ticket_no")
        for row in tickets
    )

    duplicate_message_ids = find_duplicates(
        row.get("message_id")
        for row in chats
    )

    # 只有基础主键和聊天引用都没有问题，
    # 才认为数据可以进入下一阶段。
    ready_for_next_step = not any([
        duplicate_order_nos,
        chat_order_unmatched,
        chat_ticket_unmatched,
    ])

    return {
        "records": records,

        "duplicates": {
            "order_no": duplicate_order_nos,
            "ticket_no": duplicate_ticket_nos,
            "message_id": duplicate_message_ids,
        },

        "relationships": {
            "chat_order_reference_not_found": {
                "count": len(chat_order_unmatched),
                "sample": sample_values(chat_order_unmatched),
            },
            "chat_ticket_reference_not_found": {
                "count": len(chat_ticket_unmatched),
                "sample": sample_values(chat_ticket_unmatched),
            },
            "session_with_multiple_orders": {
                "count": len(sessions_with_multiple_orders),
                "sample": sample_values(sessions_with_multiple_orders),
            },
            "session_with_multiple_tickets": {
                "count": len(sessions_with_multiple_tickets),
                "sample": sample_values(sessions_with_multiple_tickets),
            },
            "orders_without_session": {
                "count": sum(
                    1
                    for row in orders
                    if not row.get("session_id")
                ),
            },
            "tickets_without_session": {
                "count": sum(
                    1
                    for row in tickets
                    if not row.get("session_id")
                ),
            },
        },

        "ready_for_next_step": ready_for_next_step,
    }
#json输出
def json_default(value:Any)->Any:
    if isinstance(value,(datetime,date)):
        return value.isoformat()
    return str(value)
def write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    path = Path(path)

    # 自动创建父目录。
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=json_default,
                )
                + "\n"
            )