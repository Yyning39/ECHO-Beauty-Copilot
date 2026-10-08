"""
当前上下文
→ Erlangshen 判断每条消费者消息的正面/负面
→ LLM 阅读当前对话、订单、工单和历史证据
→ 规则判断风险
→ 输出统一 JSON
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
#情感分类模型
DEFAULT_SENTIMENT_MODEL = (
    "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
)
# 兼容两组常见的 OpenAI 风格环境变量。
#
# 项目原先使用：
#     LLM_BASE_URL、LLM_MODEL、LLM_API_KEY
#
# DeepSeek 或其他 OpenAI 兼容服务常使用：
#     OPENAI_API_BASE、LLM_MODEL_NAME、DEEPSEEK_API_KEY
#
# or 会从左向右选择第一个非空值，因此项目原变量的优先级更高。
DEFAULT_LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "http://127.0.0.1:11434/v1"
)
DEFAULT_LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("LLM_MODEL_NAME")
    or "qwen2.5:7b"
)
DEFAULT_LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)

# 这些词一旦出现，风险等级通常应该提高。
HIGH_RISK_KEYWORDS = (
    "投诉",
    "举报",
    "曝光",
    "维权",
    "律师",
    "媒体",
    "消协",
    "平台介入",
    "12315",
    "报警",
)
# 表示用户已经重复沟通、
# 或认为之前的问题没有得到解决的表达。
REPEAT_SERVICE_KEYWORDS = (
    "又",
    "再次",
    "一直",
    "每次",
    "反复",
    "多次",
    "说了几遍",
    "问了几次",
    "还没有",
    "怎么还",
)
# 表示物流、退款或售后处理延迟的表达。
DELAY_KEYWORDS = (
    "超过",
    "逾期",
    "延迟",
    "没到账",
    "未到账",
    "没发货",
    "未发货",
    "没有更新",
    "一直不动",
    "等了几天",
    "48小时",
    "72小时",
)

def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"上下文文件不存在：{path}"
        )
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(data, dict):
        raise ValueError(
            "上下文 JSON 的最外层必须是字典"
        )
    return data


def validate_order_binding(
    context: dict[str, Any],
    order_binding: dict[str, Any],
) -> None:
    """
    检查订单关联结果能否安全地用于当前会话分析。

    这里不重新执行订单关联，只检查两个文件是否属于同一会话，
    以及关联状态是否是 order_binding.py 定义的状态。

    如果不做这个检查，可能误把另一个会话的订单结果传给 LLM。
    """
    context_session_id = str(
        context.get("current_session_id") or ""
    ).strip()
    binding_session_id = str(
        order_binding.get("session_id") or ""
    ).strip()

    if not context_session_id:
        raise ValueError(
            "上下文缺少 current_session_id"
        )
    if not binding_session_id:
        raise ValueError(
            "订单关联结果缺少 session_id"
        )
    if context_session_id != binding_session_id:
        raise ValueError(
            "上下文与订单关联结果不属于同一会话："
            f"{context_session_id} != {binding_session_id}"
        )

    allowed_statuses = {
        "linked",
        "inferred",
        "ambiguous",
        "unbound",
        "needs_review",
    }
    binding_status = order_binding.get(
        "binding_status"
    )
    if binding_status not in allowed_statuses:
        raise ValueError(
            "未知的订单关联状态："
            f"{binding_status}"
        )

    target_order_no = order_binding.get(
        "target_order_no"
    )
    candidate_order_nos = order_binding.get(
        "candidate_order_nos",
        [],
    )
    if not isinstance(candidate_order_nos, list):
        raise ValueError(
            "candidate_order_nos 必须是列表"
        )

    # 只有 linked 或 inferred 才允许给出唯一目标订单。
    if binding_status in {"linked", "inferred"}:
        if not target_order_no:
            raise ValueError(
                f"状态 {binding_status} 缺少 target_order_no"
            )
        if target_order_no not in candidate_order_nos:
            raise ValueError(
                "target_order_no 不在 candidate_order_nos 中"
            )

    # 无法唯一确定时，target_order_no 必须为空，
    # 避免下游把某个候选误当成已确认订单。
    if binding_status in {
        "ambiguous",
        "unbound",
        "needs_review",
    } and target_order_no is not None:
        raise ValueError(
            f"状态 {binding_status} 不应设置 target_order_no"
        )


def compact_order_binding_for_llm(
    order_binding: dict[str, Any],
) -> dict[str, Any]:
    """
    只保留 LLM 判断业务范围时真正需要的订单关联字段。

    order_binding.py 的完整结果包含逐条消息核对证据。
    这些证据适合排错和审计，但同一个订单号可能重复很多次，
    全部放入 Prompt 会重复消耗 Token。

    完整关联结果仍会保存在最终输出的 raw_analysis 中；
    这里仅生成传给 LLM 的精简副本。
    """
    return {
        "binding_status": order_binding.get(
            "binding_status"
        ),
        "binding_source": order_binding.get(
            "binding_source"
        ),
        "candidate_order_nos": order_binding.get(
            "candidate_order_nos",
            [],
        ),
        "target_order_no": order_binding.get(
            "target_order_no"
        ),
        "referenced_ticket_nos": order_binding.get(
            "referenced_ticket_nos",
            [],
        ),
        "needs_human_confirmation": order_binding.get(
            "needs_human_confirmation",
            False,
        ),
        "binding_feedback": order_binding.get(
            "binding_feedback",
            "",
        ),
        "issues": order_binding.get(
            "issues",
            [],
        ),
    }

def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

def get_current_case(
    context: dict[str, Any],
) -> dict[str, Any]:
    business_context = context.get(
        "business_context",
        {},
    )
    if not isinstance(
        business_context,
        dict,
    ):
        return {}
    current_case = business_context.get(
        "current_case",
        {},
    )
    if not isinstance(
        current_case,
        dict,
    ):
        return {}
    return current_case

def get_current_messages(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    current_case = get_current_case(
        context
    )
    messages = current_case.get(
        "messages",
        [],
    )
    if not isinstance(messages, list):
        return []
    return [
        message
        for message in messages
        if isinstance(message, dict)
    ]

def is_buyer_message(
    message: dict[str, Any],
) -> bool:
    role = str(
        message.get("role") or ""
    ).strip().lower()
    buyer_roles = {
        "buyer",
        "user",
        "customer",
        "consumer",
        "消费者",
        "买家",
        "用户",
    }
    return role in buyer_roles

def load_sentiment_pipeline(
    model_name: str,
) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "当前环境没有 transformers，请先安装 transformers"
        ) from exc
    return pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
    )
def normalize_sentiment_label(
    label: str,
) -> str:

    normalized = label.strip().lower()

    if any(
        word in normalized
        for word in (
            "negative",
            "neg",
            "消极",
            "负面",
        )
    ):
        return "negative"
    if any(
        word in normalized
        for word in (
            "positive",
            "pos",
            "积极",
            "正面",
        )
    ):
        return "positive"
    return "unknown"

def predict_sentiment(
    sentiment_pipeline: Any,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        # 客服消息不参与消费者情绪判断。
        if not is_buyer_message(message):
            continue
        text = str(
            message.get("message_text") or ""
        ).strip()
        if not text:
            continue
        # 调用情感分类模型。
        # truncation=True：如果文本过长，自动截断。
        # max_length=512：限制模型最多处理的 token 数。
        raw_result = sentiment_pipeline(
            text,
            truncation=True,
            max_length=512,
        )
        if (
            isinstance(raw_result, list)
            and raw_result
            and isinstance(raw_result[0], list)
        ):
            raw_result = raw_result[0]

        if not isinstance(
            raw_result,
            list,
        ):
            continue

        if not raw_result:
            continue

        top_result = raw_result[0]

        if not isinstance(
            top_result,
            dict,
        ):
            continue

        model_label = str(
            top_result.get(
                "label",
                "unknown",
            )
        )

        try:
            score = float(
                top_result.get(
                    "score",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        results.append(
            {
                "message_id": message.get(
                    "message_id"
                ),
                "message_seq": message.get(
                    "message_seq"
                ),
                "text": text,
                "model_label": model_label,
                "polarity": normalize_sentiment_label(
                    model_label
                ),
                "score": round(
                    score,
                    6,
                ),
            }
        )
    return results

def build_llm_input(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    sentiment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    current_case = get_current_case(
        context
    )
    current_session = current_case.get(
        "session",
        {},
    )
    current_messages = current_case.get(
        "messages",
        [],
    )
    current_orders = current_case.get(
        "orders",
        [],
    )
    current_tickets = current_case.get(
        "tickets",
        [],
    )
    business_context = context.get(
        "business_context",
        {},
    )
    historical_evidence: list[
        dict[str, Any]
    ] = []

    if isinstance(
        business_context,
        dict,
    ):
        raw_history = business_context.get(
            "historical_evidence",
            [],
        )

        if isinstance(
            raw_history,
            list,
        ):
            for item in raw_history:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue
                historical_evidence.append(
                    {
                        # 检索分数和来源。
                        "retrieval": item.get(
                            "retrieval",
                            {},
                        ),
                        # 被检索到的历史聊天片段。
                        "matched_history_text": item.get(
                            "matched_history_text",
                            "",
                        ),
                        # 历史订单、工单等业务信息。
                        "business_facts": item.get(
                            "business_facts",
                            {},
                        ),
                    }
                )

    return {
        "query": context.get(
            "query"
        ),
        # 截断点告诉模型：这是某一历史时刻的回放，不可推断其后结果。
        "analysis_cutoff": context.get(
            "analysis_cutoff"
        ),
        "current_session_id": context.get(
            "current_session_id"
        ),
        "current_session": current_session,
        "current_messages": current_messages,
        "current_orders": current_orders,
        "current_tickets": current_tickets,
        # 这是 order_binding.py 对当前订单关系的核对结果。
        # LLM 应根据 binding_status 决定能否使用 target_order。
        "order_binding": compact_order_binding_for_llm(
            order_binding
        ),
        "target_order": order_binding.get(
            "target_order"
        ),
        "target_tickets": order_binding.get(
            "target_tickets",
            [],
        ),
        "sentiment_model_results": sentiment_results,
        "historical_evidence": historical_evidence,
    }

def build_system_prompt() -> str:
    """
    构造 system prompt。

    system prompt 规定：
        LLM 是谁；
        需要分析什么；
        标签有哪些；
        最终 JSON 必须长什么样。
    """
    return """
你是美妆电商人工客服的服务分析助手。

你的任务是阅读当前消费者对话、订单关联结果、当前订单、当前工单、
情感模型结果和历史服务证据，分析消费者当前的服务状态。

请严格遵守：

1. 只分析消费者的情绪，不要把客服的表达当成消费者情绪。
2. 意图必须使用固定的三级业务结构，不要自由创造同义标签：
   scene_major 表示售后大场景；scene_minor 表示具体问题；
   user_request 表示消费者此刻希望客服做什么。
3. 历史证据只能作为参考，不能把其他用户的事实当成当前用户事实。
4. 不确定时填写“无法判断”，不要编造事实。
5. 尽量使用 message_id、order_no、ticket_no 或 session_id 作为证据。
6. 只输出合法 JSON，不要输出 Markdown，不要输出解释文字。
7. 不要直接承诺退款到账、赔偿金额或具体处理结果。
8. 如果缺少信息，请放入 missing_information。
9. order_binding.binding_status 为 linked 或 inferred 时，才可以把
   target_order 和 target_tickets 当作本次问题的目标业务记录。
10. order_binding.binding_status 为 ambiguous 时，不得自行选择订单，
    必须要求确认具体订单。
11. order_binding.binding_status 为 needs_review 时，说明业务关联存在异常，
    不得根据候选订单作确定结论。
12. order_binding.binding_status 为 unbound 时，不得假设存在目标订单；
    先根据意图判断该问题是否需要订单信息。
13. missing_information 只填写当前上下文确实没有、且完成服务必须补充的信息。
    每一项必须说明需要从消费者、业务系统还是人工审核处获取。
14. known_but_not_communicated 只填写当前订单、工单或时间线中已经存在，
    但当前聊天记录尚未向消费者说明的事实，并标注事实来源。
15. recommended_actions 只填写客服、系统或人工审核下一步要执行的动作。
    “查询、同步、核实、跟进、转交”等动作不得放入 missing_information。
16. 已经存在于 current_messages、current_orders、current_tickets、
    target_order 或 target_tickets 中的信息，不得标记为 missing_information。
17. 消费者已经明确回答、接受或拒绝的事项，不得再次标记为缺失。
18. 历史案例只能用于参考处理方式，不能用它补齐当前消费者的信息。

情绪标签只能从以下集合中选择：

平静、疑惑、焦虑、不满、愤怒、失望、缓和、无法判断

情绪强度只能是：

low、medium、high、unknown

情绪趋势只能是：

stable、rising、falling、unknown

scene_major 只能从以下集合中选择：

产品咨询、补发换货、售后退货、订单服务、物流异常、退款打款、
不良反应、物流服务、会员服务、其他服务、无法判断

信息项必须尽量填写 field_code。常用 field_code 包括：

target_order、order_no、ticket_no、original_tracking_no、
reship_tracking_no、carrier、reship_carrier、logistics_status、
reship_logistics_status、damage_photo、refund_status、refund_account、
symptom_details。无法归类时可使用 other。

必须按照下面的格式返回：

{
  "intent": {
    "scene_major": "",
    "scene_minor": "",
    "user_request": "",
    "confidence": 0.0
  },
  "emotion": {
    "label": "",
    "intensity": "",
    "trend": "",
    "reason": "",
    "evidence_message_ids": []
  },
  "service_need": "",
  "missing_information": [
    {
      "field_code": "",
      "field": "",
      "reason": "",
      "required_for": "",
      "obtain_from": "customer|business_system|human_review"
    }
  ],
  "known_but_not_communicated": [
    {
      "field_code": "",
      "field": "",
      "value": "",
      "source_type": "current_order|current_ticket|timeline",
      "source_id": "",
      "reason": ""
    }
  ],
  "recommended_actions": [
    {
      "action": "",
      "owner": "customer_service|system|human_review",
      "priority": "low|medium|high",
      "basis": ""
    }
  ],
  "risk_hypotheses": [],
  "reply_strategy": "",
  "evidence": [
    {
      "source_type": "current_message|current_order|current_ticket|history",
      "source_id": "",
      "claim": ""
    }
  ]
}
""".strip()

def build_user_prompt(
    llm_input: dict[str, Any],
) -> str:
    """
    将业务上下文转换成 JSON 字符串，
    作为 LLM 的用户输入。
    json.dumps()：Python 字典 → JSON 字符串。
    ensure_ascii=False：保留中文。
    """
    return (
        "请分析下面的客服服务上下文，"
        "并严格按照要求输出 JSON：\n\n"
        + json.dumps(
            llm_input,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

def request_json(
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    request_body = json.dumps(
        body,
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    # 有些本地服务不需要 API Key。
    # 如果 api_key 为空，就不添加 Authorization。
    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    request = urllib.request.Request(
        url=url,
        data=request_body,
        method="POST",
        headers=headers,
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        response_text = response.read().decode(
            "utf-8"
        )

    data = json.loads(
        response_text
    )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "LLM 返回结果不是 JSON 对象"
        )

    return data


def extract_llm_content(
    response: dict[str, Any],
) -> str:
    choices = response.get(
        "choices",
        [],
    )

    if not isinstance(
        choices,
        list,
    ) or not choices:
        raise ValueError(
            "LLM 返回结果中没有 choices"
        )

    first_choice = choices[0]

    if not isinstance(
        first_choice,
        dict,
    ):
        raise ValueError(
            "LLM choices 格式错误"
        )

    message = first_choice.get(
        "message",
        {},
    )

    if not isinstance(
        message,
        dict,
    ):
        raise ValueError(
            "LLM message 格式错误"
        )

    content = message.get(
        "content",
        "",
    )

    # 有些多模态接口可能返回内容列表。
    if isinstance(
        content,
        list,
    ):
        content = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )

    return str(content).strip()


def parse_json_from_text(
    text: str,
) -> dict[str, Any]:
    """
    解析 LLM 返回的 JSON。

    理想情况：

        {"intent": {...}}

    但有些模型会返回：

        ```json
        {"intent": {...}}
        ```

    或者：

        分析结果如下：
        {"intent": {...}}

    所以这里做两层处理：
        1. 直接解析；
        2. 从文本中提取大括号内容。
    """
    cleaned = text.strip()

    # 删除开头的 Markdown JSON 标记。
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 删除结尾的 Markdown 代码围栏。
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        data = json.loads(
            cleaned
        )

        if isinstance(
            data,
            dict,
        ):
            return data

    except json.JSONDecodeError:
        pass

    # 从混有说明文字的结果中提取 JSON。
    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if match:
        data = json.loads(
            match.group(0)
        )

        if isinstance(
            data,
            dict,
        ):
            return data

    raise ValueError(
        "无法从 LLM 返回内容中解析 JSON"
    )


def call_llm(
    llm_input: dict[str, Any],
    base_url: str,
    model_name: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    """
    调用 LLM 进行上下文分析。

    请求地址最终会变成：

        http://127.0.0.1:11434/v1/chat/completions

    response_format 的作用是要求 LLM 返回 JSON。

    如果当前 LLM 服务不支持 response_format，
    就自动重试一次普通请求。
    """
    url = (
        base_url.rstrip("/")
        + "/chat/completions"
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                llm_input
            ),
        },
    ]

    body = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },
    }

    try:
        response = request_json(
            url=url,
            body=body,
            api_key=api_key,
            timeout=timeout,
        )

    except urllib.error.HTTPError as exc:
        # 只有请求格式不兼容时，才删除 response_format 重试。
        #
        # 401（密钥错误）、402（余额问题）、429（限流）等错误
        # 与 response_format 无关，直接抛出，避免重复产生请求。
        if exc.code not in {400, 422}:
            raise

        body.pop(
            "response_format",
            None,
        )

        response = request_json(
            url=url,
            body=body,
            api_key=api_key,
            timeout=timeout,
        )

    content = extract_llm_content(
        response
    )

    return parse_json_from_text(
        content
    )


# ============================================================
# 九、使用规则判断风险
# ============================================================

def contains_keyword(
    text: str,
    keywords: tuple[str, ...],
) -> list[str]:
    """
    找出文本中实际出现的关键词。

    例如：

        text = "我已经投诉过一次了"

    返回：

        ["投诉"]
    """
    return [
        keyword
        for keyword in keywords
        if keyword in text
    ]


def calculate_rule_risk(
    messages: list[dict[str, Any]],
    sentiment_results: list[dict[str, Any]],
    llm_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    使用规则计算风险。

    规则的职责是：

        发现明确投诉；
        发现重复服务；
        发现处理延迟；
        结合情感模型结果提高风险判断。

    规则不负责理解复杂语义，
    也不取代 LLM 的上下文分析。
    """
    buyer_texts = [
        str(
            message.get(
                "message_text"
            ) or ""
        )
        for message in messages
        if is_buyer_message(message)
    ]

    combined_text = "\n".join(
        buyer_texts
    )

    high_hits = contains_keyword(
        combined_text,
        HIGH_RISK_KEYWORDS,
    )

    repeat_hits = contains_keyword(
        combined_text,
        REPEAT_SERVICE_KEYWORDS,
    )

    delay_hits = contains_keyword(
        combined_text,
        DELAY_KEYWORDS,
    )

    negative_results = [
        item
        for item in sentiment_results
        if item.get("polarity")
        == "negative"
    ]

    latest_polarity = "unknown"

    if sentiment_results:
        latest_polarity = str(
            sentiment_results[-1].get(
                "polarity",
                "unknown",
            )
        )

    risk_score = 0
    risk_types: list[str] = []
    reasons: list[str] = []

    # 明确投诉，风险分数最高。
    if high_hits:
        risk_score += 4

        risk_types.append(
            "explicit_complaint"
        )

        reasons.append(
            "消费者出现明确投诉或升级表达："
            + "、".join(high_hits)
        )

    # 重复咨询说明服务问题可能没有被解决。
    if repeat_hits:
        risk_score += 2

        risk_types.append(
            "repeated_service"
        )

        reasons.append(
            "消费者表达了重复咨询或长期未解决："
            + "、".join(repeat_hits)
        )

    # 延迟表达说明订单或售后可能存在时效问题。
    if delay_hits:
        risk_score += 2

        risk_types.append(
            "service_delay"
        )

        reasons.append(
            "消费者表达了物流、退款或售后处理延迟："
            + "、".join(delay_hits)
        )

    # 负面情绪作为辅助风险信号。
    if negative_results:
        risk_score += 1

        risk_types.append(
            "negative_sentiment"
        )

        reasons.append(
            "情感模型在消费者消息中识别到负面情绪"
        )

    # 如果 LLM 判断情绪强度较高，
    # 也可以提高风险分数。
    llm_emotion = llm_analysis.get(
        "emotion",
        {},
    )

    if isinstance(
        llm_emotion,
        dict,
    ):
        if llm_emotion.get(
            "intensity"
        ) == "high":
            risk_score += 2

            risk_types.append(
                "high_emotion"
            )

            reasons.append(
                "LLM 判断当前情绪强度较高"
            )

    # 根据风险分数映射风险等级。
    if risk_score >= 5:
        level = "high"
    elif risk_score >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": risk_score,

        # dict.fromkeys() 可以去重，
        # 同时保留原来的顺序。
        "types": list(
            dict.fromkeys(risk_types)
        ),

        "reasons": reasons,

        "keyword_hits": {
            "high_risk": high_hits,
            "repeat_service": repeat_hits,
            "delay": delay_hits,
        },

        "latest_buyer_polarity": (
            latest_polarity
        ),
    }


# ============================================================
# 十、组织最终结果
# ============================================================

def build_fallback_llm_analysis(
    reason: str,
) -> dict[str, Any]:
    """
    当 LLM 不可用时，构造兜底结果。

    为什么需要兜底？

    因为情感模型和规则仍然可以工作。
    即使 LLM 暂时连接失败，
    也应该输出合法 JSON，
    而不是让整个 Agent 崩溃。
    """
    return {
        "intent": {
            "scene_major": "无法判断",
            "scene_minor": "无法判断",
            "user_request": "无法判断",
            "confidence": 0.0,
        },
        "emotion": {
            "label": "无法判断",
            "intensity": "unknown",
            "trend": "unknown",
            "reason": reason,
            "evidence_message_ids": [],
        },
        "service_need": "",
        "missing_information": [
            {
                "field_code": "other",
                "field": "LLM 上下文分析结果",
                "reason": reason,
                "required_for": "生成细粒度意图、情绪和服务建议",
                "obtain_from": "human_review",
            }
        ],
        "known_but_not_communicated": [],
        "recommended_actions": [
            {
                "action": "人工检查当前会话并补充分析",
                "owner": "human_review",
                "priority": "high",
                "basis": reason,
            }
        ],
        "risk_hypotheses": [],
        "reply_strategy": "",
        "evidence": [],
    }


def normalize_structured_list(
    value: Any,
    primary_key: str,
) -> list[dict[str, Any]]:
    """
    将 LLM 返回的列表统一成“字典列表”。

    新版 Prompt 要求每一项都是字典，但模型或旧结果仍可能返回字符串。
    对字符串进行兼容包装，能够保证后续前端始终读取同一种结构。

    primary_key：
        missing_information 使用 field；
        known_but_not_communicated 使用 field；
        recommended_actions 使用 action。
    """
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
            continue
        if isinstance(item, str) and item.strip():
            normalized.append(
                {
                    primary_key: item.strip(),
                }
            )
    return normalized


def append_unique_item(
    items: list[dict[str, Any]],
    item: dict[str, Any],
    identity_key: str,
) -> None:
    """
    根据一个核心字段去重后追加结构化项目。

    例如已经存在 field='目标订单' 时，
    Python 规则就不再追加第二条同名缺失信息。
    """
    identity_value = item.get(identity_key)
    if any(
        existing.get(identity_key) == identity_value
        for existing in items
    ):
        return
    items.append(item)


ALLOWED_SCENE_MAJORS = {
    "产品咨询",
    "补发换货",
    "售后退货",
    "订单服务",
    "物流异常",
    "退款打款",
    "不良反应",
    "物流服务",
    "会员服务",
    "其他服务",
    "无法判断",
}


def normalize_intent(value: Any) -> dict[str, Any]:
    """把 LLM 的意图结果固定成赛事业务可统计的结构。

    scene_major/scene_minor 用来做看板统计，user_request 保留消费者这一次
    希望客服执行的动作。函数也兼容旧版 categories/primary_category 输出。
    """

    if not isinstance(value, dict):
        value = {}

    scene_major = str(
        value.get("scene_major")
        or value.get("primary_category")
        or "无法判断"
    ).strip()
    old_categories = value.get("categories", [])
    category_text = " ".join(
        str(item)
        for item in old_categories
        if item
    ) if isinstance(old_categories, list) else ""

    # 兼容之前已经保存的自由标签，避免升级脚本后旧结果无法展示。
    if scene_major not in ALLOWED_SCENE_MAJORS:
        combined = f"{scene_major} {category_text}"
        if any(word in combined for word in ("换货", "补发", "破损")):
            scene_major = "补发换货"
        elif "退货" in combined:
            scene_major = "售后退货"
        elif "退款" in combined:
            scene_major = "退款打款"
        elif any(word in combined for word in ("物流", "快递")):
            scene_major = "物流异常"
        elif any(word in combined for word in ("过敏", "不良反应")):
            scene_major = "不良反应"
        else:
            scene_major = "其他服务"

    scene_minor = str(value.get("scene_minor") or "").strip()
    if not scene_minor and isinstance(old_categories, list):
        # 优先选择比大类更具体的旧标签。
        scene_minor = next(
            (
                str(item)
                for item in old_categories
                if item and str(item) != scene_major
            ),
            scene_major,
        )
    if not scene_minor:
        scene_minor = scene_major

    user_request = str(value.get("user_request") or "").strip()
    if not user_request:
        request_words = [
            str(item)
            for item in old_categories
            if item and any(
                word in str(item)
                for word in ("催促", "查询", "申请", "咨询", "确认")
            )
        ] if isinstance(old_categories, list) else []
        user_request = request_words[0] if request_words else scene_minor

    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "scene_major": scene_major,
        "scene_minor": scene_minor,
        "user_request": user_request,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def infer_field_code(item: dict[str, Any]) -> str:
    """根据中文字段名补充稳定的 field_code，供规则做精确比较。"""

    existing = str(item.get("field_code") or "").strip()
    if existing:
        return existing

    field = str(item.get("field") or "")
    mappings = (
        (("补发物流单号", "补发单号"), "reship_tracking_no"),
        (("原订单物流单号", "原物流单号"), "original_tracking_no"),
        (("补发快递", "补发承运"), "reship_carrier"),
        (("补发物流状态", "补发物流进度", "补发轨迹"), "reship_logistics_status"),
        (("物流状态", "物流进度", "物流轨迹"), "logistics_status"),
        (("快递公司", "承运商"), "carrier"),
        (("发货仓库", "仓库"), "warehouse"),
        (("破损图片", "破损照片", "商品图片"), "damage_photo"),
        (("退款状态", "退款进度"), "refund_status"),
        (("退款账户", "退款账号"), "refund_account"),
        (("症状", "不良反应详情"), "symptom_details"),
        (("具体订单", "目标订单"), "target_order"),
        (("订单号",), "order_no"),
        (("工单号",), "ticket_no"),
    )
    for keywords, field_code in mappings:
        if any(keyword in field for keyword in keywords):
            return field_code
    return "other"


def collect_known_business_facts(
    order_binding: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """从已确认的目标订单/工单中建立可核验的事实表。"""

    facts: dict[str, dict[str, Any]] = {}
    order = order_binding.get("target_order")
    if isinstance(order, dict):
        order_no = order.get("order_no")
        order_fields = {
            "order_no": order_no,
            "original_tracking_no": order.get("tracking_no"),
            "carrier": order.get("carrier"),
        }
        for field_code, value in order_fields.items():
            if value not in (None, ""):
                facts[field_code] = {
                    "value": value,
                    "source_type": "current_order",
                    "source_id": order_no,
                }

    for ticket in order_binding.get("target_tickets", []):
        if not isinstance(ticket, dict):
            continue
        ticket_no = ticket.get("ticket_no")
        if ticket_no:
            facts["ticket_no"] = {
                "value": ticket_no,
                "source_type": "current_ticket",
                "source_id": ticket_no,
            }
        detail = ticket.get("detail", {})
        if not isinstance(detail, dict):
            detail = {}
        detail_fields = {
            "original_tracking_no": detail.get("原订单物流单号"),
            "reship_tracking_no": detail.get("补发物流单号"),
            "reship_carrier": detail.get("快递公司"),
            "warehouse": detail.get("发货仓库"),
            "refund_status": detail.get("退款状态"),
        }
        for field_code, value in detail_fields.items():
            if value not in (None, ""):
                facts[field_code] = {
                    "value": value,
                    "source_type": "current_ticket",
                    "source_id": ticket_no,
                }
    return facts


def validate_information_outputs(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    intent: dict[str, Any],
    missing_information: list[dict[str, Any]],
    known_but_not_communicated: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """用结构化业务事实校验 LLM 提出的信息项。

    LLM 擅长理解“需要什么”，但不应独自判断数据库里“有没有”。这里让
    LLM 提候选、Python 查订单/工单事实，形成可解释的混合方案。
    """

    facts = collect_known_business_facts(order_binding)
    current_messages = get_current_messages(context)
    communicated_text = "\n".join(
        str(message.get("message_text") or "")
        for message in current_messages
    )
    customer_facing_codes = {
        "order_no",
        "ticket_no",
        "original_tracking_no",
        "reship_tracking_no",
        "carrier",
        "reship_carrier",
        "logistics_status",
        "reship_logistics_status",
        "refund_status",
    }
    removed_missing: list[dict[str, Any]] = []
    removed_known: list[dict[str, Any]] = []
    checked_missing: list[dict[str, Any]] = []
    checked_known: list[dict[str, Any]] = []

    for item in missing_information:
        normalized = dict(item)
        field_code = infer_field_code(normalized)
        normalized["field_code"] = field_code
        fact = facts.get(field_code)
        if fact is None:
            append_unique_item(
                checked_missing,
                normalized,
                "field_code",
            )
            continue

        # 数据库已经有值，所以它不能继续出现在“确实缺少”中。
        removed_missing.append(normalized)
        value_text = str(fact.get("value") or "")
        if (
            field_code in customer_facing_codes
            and value_text
            and value_text not in communicated_text
        ):
            append_unique_item(
                checked_known,
                {
                    "field_code": field_code,
                    "field": normalized.get("field", field_code),
                    "value": fact.get("value"),
                    "source_type": fact.get("source_type"),
                    "source_id": fact.get("source_id"),
                    "reason": "结构化业务数据已有该事实，但当前聊天尚未告知消费者",
                },
                "field_code",
            )

    for item in known_but_not_communicated:
        normalized = dict(item)
        field_code = infer_field_code(normalized)
        normalized["field_code"] = field_code
        fact = facts.get(field_code)
        # 仓库等内部信息不属于必须主动告诉消费者的服务事实。
        if field_code not in customer_facing_codes or fact is None:
            removed_known.append(normalized)
            continue
        value_text = str(fact.get("value") or "")
        if value_text and value_text in communicated_text:
            removed_known.append(normalized)
            continue
        normalized["value"] = fact.get("value")
        normalized["source_type"] = fact.get("source_type")
        normalized["source_id"] = fact.get("source_id")
        append_unique_item(
            checked_known,
            normalized,
            "field_code",
        )

    # 对“催补发/问进度”的场景，只有运单号还不够；当前物流轨迹是回答
    # “发到哪里了”的必要事实。订单/工单快照没有该字段时明确标成系统待查。
    request_text = str(intent.get("user_request") or "")
    if (
        intent.get("scene_major") == "补发换货"
        and any(word in request_text for word in ("催", "进度", "发出", "到哪"))
        and "reship_logistics_status" not in facts
    ):
        append_unique_item(
            checked_missing,
            {
                "field_code": "reship_logistics_status",
                "field": "补发件当前物流状态",
                "reason": "当前订单和工单只有补发单号，没有实时物流轨迹",
                "required_for": "准确回复补发件是否发出及当前所在节点",
                "obtain_from": "business_system",
            },
            "field_code",
        )

    return checked_missing, checked_known, {
        "removed_missing_items": removed_missing,
        "removed_known_items": removed_known,
        "known_fact_codes": sorted(facts),
    }


def build_final_result(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    sentiment_model_name: str,
    sentiment_results: list[dict[str, Any]],
    llm_analysis: dict[str, Any],
    llm_status: str,
    llm_error: str | None,
    rule_risk: dict[str, Any],
) -> dict[str, Any]:
    """
    组织最终输出。

    final：
        给前端和后续模块直接使用。

    raw_analysis：
        保存每个模型的原始分析结果，
        便于后续排错和评估。
    """
    emotion = llm_analysis.get(
        "emotion",
        {},
    )

    if not isinstance(
        emotion,
        dict,
    ):
        emotion = {}

    binding_status = order_binding.get(
        "binding_status"
    )
    intent = normalize_intent(
        llm_analysis.get("intent", {})
    )
    missing_information = normalize_structured_list(
        llm_analysis.get(
            "missing_information",
            [],
        ),
        primary_key="field",
    )

    known_but_not_communicated = normalize_structured_list(
        llm_analysis.get(
            "known_but_not_communicated",
            [],
        ),
        primary_key="field",
    )

    recommended_actions = normalize_structured_list(
        llm_analysis.get(
            "recommended_actions",
            [],
        ),
        primary_key="action",
    )

    (
        missing_information,
        known_but_not_communicated,
        information_validation,
    ) = validate_information_outputs(
        context=context,
        order_binding=order_binding,
        intent=intent,
        missing_information=missing_information,
        known_but_not_communicated=known_but_not_communicated,
    )

    binding_missing_information: dict[str, Any] | None = None
    if binding_status == "ambiguous":
        binding_missing_information = {
            "field_code": "target_order",
            "field": "本次问题对应的具体订单",
            "reason": "当前会话存在多笔候选订单，无法唯一确定目标订单",
            "required_for": "查询正确的订单、工单和售后进度",
            "obtain_from": "customer",
        }
    elif binding_status == "needs_review":
        binding_missing_information = {
            "field_code": "target_order",
            "field": "正确的订单与工单关联关系",
            "reason": "当前订单或工单关联存在未匹配引用或冲突",
            "required_for": "避免使用错误业务记录生成结论",
            "obtain_from": "human_review",
        }

    if binding_missing_information:
        append_unique_item(
            items=missing_information,
            item=binding_missing_information,
            identity_key="field_code",
        )

    return {
        "schema_version": (
            "agent_analysis_v3"
        ),

        "session_id": context.get(
            "current_session_id"
        ),

        "query": context.get(
            "query"
        ),

        # 前端主要读取这个字段。
        "final": {
            "intent": intent,

            "emotion": {
                **emotion,

                # 将情感模型的正负面结果
                # 和 LLM 的细粒度情绪同时保存。
                "sentiment_polarity": (
                    rule_risk.get(
                        "latest_buyer_polarity",
                        "unknown",
                    )
                ),
            },

            "service_need": llm_analysis.get(
                "service_need",
                "",
            ),

            "risk": rule_risk,

            "reply_strategy": llm_analysis.get(
                "reply_strategy",
                "",
            ),

            # 三类结果分别驱动追问、信息同步和客服下一步操作。
            "missing_information": missing_information,
            "known_but_not_communicated": (
                known_but_not_communicated
            ),
            "recommended_actions": recommended_actions,

            # 前端通过这部分判断能否直接展示目标订单，
            # 或者是否应提示客服进行人工确认。
            "order_binding": {
                "binding_status": binding_status,
                "binding_source": order_binding.get(
                    "binding_source"
                ),
                "target_order_no": order_binding.get(
                    "target_order_no"
                ),
                "candidate_order_nos": order_binding.get(
                    "candidate_order_nos",
                    [],
                ),
                "referenced_ticket_nos": order_binding.get(
                    "referenced_ticket_nos",
                    [],
                ),
                "needs_human_confirmation": order_binding.get(
                    "needs_human_confirmation",
                    False,
                ),
                "binding_feedback": order_binding.get(
                    "binding_feedback",
                    "",
                ),
            },
        },

        # 这里保存详细调试信息。
        "raw_analysis": {
            "sentiment_model": {
                "model_name": sentiment_model_name,
                "message_results": sentiment_results,
            },

            "llm": {
                "status": llm_status,
                "error": llm_error,
                "analysis": llm_analysis,
            },

            "rule_risk": rule_risk,

            # 记录哪些 LLM 候选被结构化事实纠正，便于评估和排错。
            "information_validation": information_validation,

            # 保存完整关联结果，方便排查某个订单为何被选中。
            "order_binding": order_binding,
        },

        "evidence": llm_analysis.get(
            "evidence",
            [],
        ),

        # 保留顶层字段，便于后续脚本直接读取；
        # final 中也保存同样内容，供前端统一展示。
        "missing_information": missing_information,
        "known_but_not_communicated": (
            known_but_not_communicated
        ),
        "recommended_actions": recommended_actions,
    }


# ============================================================
# 十一、主程序
# ============================================================

def main() -> None:
    """
    主程序负责调度各个步骤。

    主程序本身不负责具体分析，
    只负责按照正确顺序调用函数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze customer service context."
        )
    )

    parser.add_argument(
        "--context-file",
        type=Path,
        required=True,
        help=(
            "run_retrieval_context.py "
            "输出的 JSON 文件"
        ),
    )

    parser.add_argument(
        "--order-binding",
        type=Path,
        required=True,
        help=(
            "order_binding.py 输出的订单关联 JSON 文件"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="分析结果输出 JSON 文件",
    )

    parser.add_argument(
        "--sentiment-model",
        type=str,
        default=DEFAULT_SENTIMENT_MODEL,
        help="情感分类模型名称",
    )

    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help="OpenAI 兼容 LLM 服务地址",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM 模型名称",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_LLM_API_KEY,
        help="LLM API Key",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="LLM 请求超时时间，单位为秒",
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help=(
            "跳过 LLM，"
            "只运行情感模型和规则"
        ),
    )

    args = parser.parse_args()

    # 第一步：读取 RAG 上下文。
    context = read_json(
        args.context_file
    )

    # 第二步：读取订单关联结果，并确认它属于当前会话。
    order_binding = read_json(
        args.order_binding
    )
    validate_order_binding(
        context=context,
        order_binding=order_binding,
    )

    # 第三步：获取当前聊天记录。
    messages = get_current_messages(
        context
    )

    print(
        f"当前会话消息数：{len(messages)}"
    )

    # 第四步：加载情感模型。
    print(
        "正在加载情感模型："
        f"{args.sentiment_model}"
    )

    sentiment_pipeline = (
        load_sentiment_pipeline(
            args.sentiment_model
        )
    )

    # 第五步：分析消费者消息情绪。
    sentiment_results = predict_sentiment(
        sentiment_pipeline=sentiment_pipeline,
        messages=messages,
    )

    print(
        "消费者消息情感分析完成："
        f"{len(sentiment_results)} 条"
    )

    # 第六步：组织 LLM 输入。
    llm_input = build_llm_input(
        context=context,
        order_binding=order_binding,
        sentiment_results=sentiment_results,
    )

    llm_analysis: dict[str, Any] = {}
    llm_status = "skipped"
    llm_error: str | None = None

    # 第七步：调用 LLM。
    if args.skip_llm:
        llm_analysis = build_fallback_llm_analysis(
            reason="本次运行跳过了 LLM"
        )

    else:
        try:
            print(
                "正在调用 LLM："
                f"{args.llm_model}"
            )

            llm_analysis = call_llm(
                llm_input=llm_input,
                base_url=args.llm_base_url,
                model_name=args.llm_model,
                api_key=args.api_key,
                timeout=args.timeout,
            )

            llm_status = "success"

        except Exception as exc:
            # LLM 出错后不让整个程序崩溃。
            llm_status = "failed"
            llm_error = str(exc)

            llm_analysis = (
                build_fallback_llm_analysis(
                    reason="LLM 调用失败"
                )
            )

            print(
                f"LLM 调用失败：{exc}"
            )

    # 第八步：根据情感模型、当前聊天和 LLM 结果，
    # 运行确定性风险规则。
    rule_risk = calculate_rule_risk(
        messages=messages,
        sentiment_results=sentiment_results,
        llm_analysis=llm_analysis,
    )

    # 第九步：合并最终结果。
    result = build_final_result(
        context=context,
        order_binding=order_binding,
        sentiment_model_name=args.sentiment_model,
        sentiment_results=sentiment_results,
        llm_analysis=llm_analysis,
        llm_status=llm_status,
        llm_error=llm_error,
        rule_risk=rule_risk,
    )

    # 第十步：如果用户提供了输出路径，
    # 就将结果保存到 JSON 文件。
    if args.output is not None:
        save_json(
            path=args.output,
            data=result,
        )

        print(
            f"分析结果已保存：{args.output}"
        )

    # 第十一步：同时在终端打印结果。
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
