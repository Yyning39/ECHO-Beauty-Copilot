#判断消费者消息对应哪个订单。
"""
输入：run_retrieval_context.py 生成的上下文 JSON。
输出：1. 候选订单。
2. 能否唯一确定订单。
3. 关联依据。
4. 未匹配引用或冲突。
5. 对应订单和工单的业务事实。

linked 表示数据关联明确，
不表示消费者已经通过卡片或按钮确认过订单。

本脚本使用当前会话的事实，
不使用 RAG 检索出来的相似历史案例决定订单归属。
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_context_file(path: Path) -> dict[str, Any]:
    # is_file() 判断路径是否指向一个存在的文件。
    if not path.is_file():
        raise FileNotFoundError(
            f"上下文文件不存在：{path}"
        )
    text = path.read_text(
        encoding="utf-8-sig"
    )
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(
            "上下文 JSON 的最外层必须是字典。"
        )

    return data


def extract_current_case(
    context: dict[str, Any],
) -> dict[str, Any]:

    # get() 从字典中取值。
    # key 不存在时返回 None，不会立即抛出 KeyError。
    business_context = context.get(
        "business_context"
    )

    if not isinstance(business_context, dict):
        raise ValueError(
            "上下文缺少 business_context 字典。"
        )

    case = business_context.get(
        "current_case"
    )

    if not isinstance(case, dict):
        raise ValueError(
            "上下文缺少 business_context.current_case。"
        )

    session = case.get("session")

    if not isinstance(session, dict):
        raise ValueError(
            "current_case.session 必须是字典。"
        )

    session_id = normalize_id(
        session.get("session_id")
    )

    if not session_id:
        raise ValueError(
            "current_case.session 缺少 session_id。"
        )

    # 上下文最外层也有 current_session_id。
    # 如果存在，就检查它是否和 current_case 一致。
    outer_session_id = normalize_id(
        context.get("current_session_id")
    )

    if (
        outer_session_id
        and outer_session_id != session_id
    ):
        raise ValueError(
            "current_session_id 与 current_case 的会话 ID 不一致。"
        )

    # for 会依次把 field_name 赋值为这三个字符串。
    for field_name in (
        "messages",
        "orders",
        "tickets",
    ):
        records = case.get(field_name)

        if not isinstance(records, list):
            raise ValueError(
                f"current_case.{field_name} 必须是列表。"
            )

        for position, record in enumerate(
            records,
            start=1,
        ):
            if not isinstance(record, dict):
                raise ValueError(
                    f"{field_name} 的第 {position} 条记录必须是字典。"
                )

    for message in case["messages"]:
        # 判断 key 是否存在，和判断值是否为空是两回事。
        #
        # 存在且为 None：
        # 上游保留了字段，但这条消息没有关联编号。
        #
        # key 不存在：
        # 可能正在读取修改 compact_case() 之前的旧文件。
        for field_name in (
            "related_order_no",
            "related_ticket_no",
        ):
            if field_name not in message:
                raise ValueError(
                    f"消息缺少字段 {field_name}。"
                    "请用修改后的 business_context.py "
                    "重新生成上下文 JSON。"
                )

    return case


def build_lookup(
    records: list[dict[str, Any]],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    """
    目的：把记录列表转成按编号查找的字典。
    """

    # {} 创建空字典。
    lookup: dict[str, dict[str, Any]] = {}

    for position, record in enumerate(
        records,
        start=1,
    ):
        record_id = normalize_id(
            record.get(id_field)
        )

        if not record_id:
            raise ValueError(
                f"{id_field} 列表的第 {position} 条记录缺少编号。"
            )

        # 对字典使用 in，检查的是 key 是否存在。
        if record_id in lookup:
            raise ValueError(
                f"{id_field} 出现重复编号：{record_id}"
            )

        # dict(record) 创建浅拷贝。
        # 后面统一编号时，不直接修改调用方传入的原字典。
        normalized_record = dict(record)
        normalized_record[id_field] = record_id

        # 对工单再统一它关联的订单编号。
        if id_field == "ticket_no":
            normalized_record["order_no"] = normalize_id(
                record.get("order_no")
            )

        # 将完整记录存到编号对应的位置。
        lookup[record_id] = normalized_record

    return lookup


def inspect_message_relations(
    messages: list[dict[str, Any]],
    orders_by_no: dict[str, dict[str, Any]],
    tickets_by_no: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    目的：
    逐条检查消息关联，收集候选订单、证据和异常。
    """

    # set() 创建空集合。
    # 集合适合去重，多条消息关联订单 A，集合中仍只有一个 A。
    # 注意：{} 是空字典，不是空集合。
    candidate_order_nos: set[str] = set()
    referenced_ticket_nos: set[str] = set()

    # [] 创建列表。
    # 列表保存逐条证据和问题，使用 append() 添加。
    evidence: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    # 区分“完全没有提供关联”和“提供了关联但无法匹配”。
    has_explicit_reference = False

    for message in messages:
        message_id = message.get("message_id")

        direct_order_no = normalize_id(
            message.get("related_order_no")
        )

        ticket_no = normalize_id(
            message.get("related_ticket_no")
        )

        # 只要任意关联字段非空，就说明这条消息提供了引用。
        if direct_order_no or ticket_no:
            has_explicit_reference = True

        # 保存从工单读取到的订单号。
        # 每次循环重新设为空，避免沿用上一条消息的值。
        ticket_order_no = ""

        # 路径一：消息直接关联订单。
        if direct_order_no:
            if direct_order_no in orders_by_no:
                candidate_order_nos.add(
                    direct_order_no
                )

                evidence.append(
                    {
                        "message_id": message_id,
                        "source_field": "related_order_no",
                        "order_no": direct_order_no,
                    }
                )
            else:
                # 找不到不能直接解释成 Excel 错误。
                # 也可能是上游没有加载这笔订单。
                issues.append(
                    {
                        "type": "unmatched_order_reference",
                        "message_id": message_id,
                        "order_no": direct_order_no,
                        "reason": "消息关联的订单不在当前会话输入订单中。",
                    }
                )

        # 路径二：消息关联工单，再通过工单关联订单。
        if ticket_no:
            # 工单号非空就记录，找不到的引用也保留用于排查。
            referenced_ticket_nos.add(
                ticket_no
            )

            # dict.get(key) 找不到时返回 None。
            ticket = tickets_by_no.get(
                ticket_no
            )

            if ticket is None:
                issues.append(
                    {
                        "type": "unmatched_ticket_reference",
                        "message_id": message_id,
                        "ticket_no": ticket_no,
                        "reason": "消息关联的工单不在当前会话输入工单中。",
                    }
                )
            else:
                ticket_order_no = normalize_id(
                    ticket.get("order_no")
                )

                # 工单存在，但可能没有关联订单。
                # 空值本身不等于错误，仍保留工单关联证据。
                evidence.append(
                    {
                        "message_id": message_id,
                        "source_field": "related_ticket_no",
                        "ticket_no": ticket_no,
                        "order_no": ticket_order_no or None,
                    }
                )

                if ticket_order_no:
                    if ticket_order_no in orders_by_no:
                        candidate_order_nos.add(
                            ticket_order_no
                        )
                    else:
                        issues.append(
                            {
                                "type": "unmatched_ticket_order",
                                "message_id": message_id,
                                "ticket_no": ticket_no,
                                "order_no": ticket_order_no,
                                "reason": "工单关联的订单不在当前会话输入订单中。",
                            }
                        )

        # 两条路径都给出了编号时，检查编号是否一致。
        #
        # 即使某个订单记录没有加载，
        # 仍然可以比较两个编号是否相同。
        if (
            direct_order_no
            and ticket_order_no
            and direct_order_no != ticket_order_no
        ):
            issues.append(
                {
                    "type": "message_ticket_order_conflict",
                    "message_id": message_id,
                    "ticket_no": ticket_no,
                    "message_order_no": direct_order_no,
                    "ticket_order_no": ticket_order_no,
                    "reason": "消息关联订单与工单关联订单不一致。",
                }
            )

    return {
        # sorted() 把集合转换成排序后的列表。
        # 集合不能直接通过 json.dumps() 保存，
        # 转成列表也让每次输出顺序稳定。
        "candidate_order_nos": sorted(
            candidate_order_nos
        ),
        "referenced_ticket_nos": sorted(
            referenced_ticket_nos
        ),
        "has_explicit_reference": has_explicit_reference,
        "evidence": evidence,
        "issues": issues,
    }


def build_order_binding(
    current_case: dict[str, Any],
) -> dict[str, Any]:
    """
    目的：
    汇总中间结果，判断最终关联状态。

    内部业务顺序：
    1. 建立查找表。
    2. 检查消息关系。
    3. 必要时使用会话订单兜底。
    4. 根据异常数量、候选数量判断状态。
    5. 组织下游需要的业务事实和证据。

    兜底原则：
    消息没有形成订单候选且核对没有异常时，
    才考虑当前会话的订单。

    有错误引用时不自动改绑到另一笔订单。
    """

    session_id = normalize_id(
        current_case["session"].get(
            "session_id"
        )
    )

    orders_by_no = build_lookup(
        records=current_case["orders"],
        id_field="order_no",
    )

    tickets_by_no = build_lookup(
        records=current_case["tickets"],
        id_field="ticket_no",
    )

    inspected = inspect_message_relations(
        messages=current_case["messages"],
        orders_by_no=orders_by_no,
        tickets_by_no=tickets_by_no,
    )

    candidate_order_nos = inspected[
        "candidate_order_nos"
    ]
    evidence = inspected["evidence"]
    issues = inspected["issues"]

    # binding_source 说明“候选来自哪里”，
    # binding_status 说明“候选能否可靠地唯一确定”。
    # 两个字段回答的问题不同。
    binding_source = (
        "message_relation"
        if candidate_order_nos
        else "none"
    )

    # not list：列表为空时成立。
    #
    # 如果没有从消息得到订单候选，且没有异常，
    # 就使用当前会话订单列表作为兜底候选。
    #
    # 工单有引用但没有订单号，也可能进入这个兜底。
    if not candidate_order_nos and not issues:
        candidate_order_nos = sorted(
            orders_by_no.keys()
        )

        if candidate_order_nos:
            binding_source = "session_orders"

            for order_no in candidate_order_nos:
                evidence.append(
                    {
                        "session_id": session_id,
                        "source_field": "current_case.orders",
                        "order_no": order_no,
                    }
                )

    # 初始值设为 None。
    # 只有能够可靠地唯一确定时，才填写目标订单。
    target_order_no: str | None = None

    # 判断顺序很重要：
    # 先检查异常，再检查候选数量。
    # 有一个候选不代表一定可靠，还可能同时存在冲突。
    if issues:
        binding_status = "needs_review"
        needs_human_confirmation = True
        binding_feedback = (
            "存在未匹配引用或关联冲突，"
            "需要检查数据加载范围或由人工确认。"
        )

    elif len(candidate_order_nos) > 1:
        binding_status = "ambiguous"
        needs_human_confirmation = True
        binding_feedback = (
            "当前会话涉及多笔候选订单，"
            "尚未唯一确定本次问题对应的订单。"
        )

    elif len(candidate_order_nos) == 1:
        # 列表只有一个元素时，索引 0 就是唯一候选。
        target_order_no = candidate_order_nos[0]
        needs_human_confirmation = False

        if binding_source == "message_relation":
            binding_status = "linked"
            binding_feedback = (
                "消息中的业务关联明确指向一笔订单，"
                "且已通过当前输入记录核对。"
            )
        else:
            binding_status = "inferred"
            binding_feedback = (
                "消息未提供可用订单关联，"
                "依据当前会话唯一订单进行兜底。"
            )

    else:
        binding_status = "unbound"

        # 没有订单不一定需要打断服务询问消费者。
        # 是否必须确认订单，要结合后续意图分析：
        # 售前产品咨询可以无订单；
        # 查询退款进度通常需要订单或工单。
        needs_human_confirmation = False
        binding_feedback = (
            "当前输入没有订单关联依据。"
            "是否需要补充订单，应结合用户意图判断。"
        )

    # target_order_no 为 None 时，目标订单事实也为 None。
    target_order = (
        orders_by_no[target_order_no]
        if target_order_no is not None
        else None
    )

    # candidate_order_nos 仅包含已在查找表中核对过的编号。
    # 所以这里可以使用 [] 精确取值。
    candidate_orders = [
        orders_by_no[order_no]
        for order_no in candidate_order_nos
    ]

    # 单独收集属于目标订单的全部当前会话工单。
    # 这些工单可以帮助下游理解处理进度。
    #
    # 如果没有唯一目标订单，则不自动选出目标工单。
    target_tickets = [
        ticket
        for ticket in tickets_by_no.values()
        if (
            target_order_no is not None
            and ticket.get("order_no") == target_order_no
        )
    ]

    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "scope": "current_case",
        "binding_status": binding_status,
        "binding_source": binding_source,
        "candidate_order_nos": candidate_order_nos,
        "target_order_no": target_order_no,
        "referenced_ticket_nos": inspected[
            "referenced_ticket_nos"
        ],
        "has_explicit_reference": inspected[
            "has_explicit_reference"
        ],
        "needs_human_confirmation": needs_human_confirmation,
        "binding_feedback": binding_feedback,

        # 给后续 LLM 或规则模块使用的业务事实。
        "candidate_orders": candidate_orders,
        "target_order": target_order,
        "target_tickets": target_tickets,

        # 保留关联依据以及核对失败的具体原因。
        "evidence": evidence,
        "issues": issues,
    }


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    目的：
    将 Python 字典保存成 JSON 文件。

    此函数只负责保存，
    不参与订单判断。
    """

    # path.parent 是文件所在目录。
    # parents=True：缺少多级父目录时一起创建。
    # exist_ok=True：目录已经存在时不报错。
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # dumps()：Python 对象 -> JSON 字符串。
    # ensure_ascii=False：中文直接显示，不转成 \\uXXXX。
    # indent=2：每层缩进两个空格，方便阅读。
    text = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    # 写入指定文件；同路径已有文件会被覆盖。
    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    """
    目的：
    调度整个脚本，不在这里堆积关联判断规则。

    流程：
    解析参数 -> 读取文件 -> 提取当前会话
    -> 执行关联 -> 保存和打印结果。
    """

    parser = argparse.ArgumentParser(
        description="核对当前会话中的订单和工单关联。"
    )

    parser.add_argument(
        "--context-file",
        type=Path,
        required=True,
        help="run_retrieval_context.py 输出的上下文 JSON。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "analysis"
            / "order_binding_result.json"
        ),
        help="订单关联结果的输出路径。",
    )

    # parse_args() 读取运行命令中的参数。
    # --context-file 会变成 args.context_file。
    # 命令行名称中的连字符，转成属性名中的下划线。
    args = parser.parse_args()

    context = read_context_file(
        args.context_file
    )

    current_case = extract_current_case(
        context
    )

    result = build_order_binding(
        current_case
    )

    save_json(
        path=args.output,
        data=result,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\n订单关联结果已保存：{args.output}"
    )


# 直接执行 python scripts/order_binding.py 时运行 main()。
# 被其他脚本 import 时，不会自动执行 main()。
# 这样以后可以直接导入 build_order_binding() 使用。
if __name__ == "__main__":
    main()