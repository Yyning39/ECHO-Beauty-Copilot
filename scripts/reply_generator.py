"""根据 Agent 分析结果生成“人工客服可审核”的回复草稿。

业务定位：
    这个项目是人工客服辅助插件，不是自动回复机器人。因此本脚本只生成草稿、
    依据和风险提示；最终是否发送、如何修改，始终由人工客服决定。

输入：
    1. run_retrieval_context.py 输出的上下文；
    2. order_binding.py 输出的订单关联结果；
    3. agent_analysis.py 输出的分析结果。

输出：
    回复草稿、引用事实、客服下一步动作、禁止承诺、人工审核原因。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
from pathlib import Path
from typing import Any

from agent_analysis import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    extract_llm_content,
    infer_field_code,
    parse_json_from_text,
    request_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象，并在输入格式错误时尽早报错。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 最外层必须是对象：{path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    """把结果保存为 UTF-8 JSON，便于中文直接阅读。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def validate_same_session(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    """确保三个输入属于同一会话，防止串单。"""

    session_ids = {
        str(context.get("current_session_id") or "").strip(),
        str(order_binding.get("session_id") or "").strip(),
        str(analysis.get("session_id") or "").strip(),
    }
    session_ids.discard("")
    if len(session_ids) != 1:
        raise ValueError(
            "上下文、订单关联和分析结果不属于同一会话："
            f"{sorted(session_ids)}"
        )
    return next(iter(session_ids))


def build_grounding_package(
    order_binding: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """只收集允许写入回复的已核验事实和待办。

    为什么不把整个上下文再次交给回复模型：回复阶段只需要“已经确定的结论”，
    输入越小越便宜，也越不容易把其他用户的历史案例误写进当前回复。
    """

    final = analysis.get("final", {})
    if not isinstance(final, dict):
        final = {}

    raw_known_facts = final.get("known_but_not_communicated", [])
    raw_missing_information = final.get("missing_information", [])
    recommended_actions = final.get("recommended_actions", [])

    # 兼容还没有 field_code 的旧版分析结果。新版会直接携带该字段。
    known_facts: list[dict[str, Any]] = []
    customer_facing_codes = {
        "order_no", "ticket_no", "original_tracking_no",
        "reship_tracking_no", "carrier", "reship_carrier",
        "logistics_status", "reship_logistics_status", "refund_status",
    }
    if isinstance(raw_known_facts, list):
        for item in raw_known_facts:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["field_code"] = infer_field_code(normalized)
            if normalized["field_code"] in customer_facing_codes:
                known_facts.append(normalized)

    missing_information: list[dict[str, Any]] = []
    if isinstance(raw_missing_information, list):
        for item in raw_missing_information:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["field_code"] = infer_field_code(normalized)
            missing_information.append(normalized)

    return {
        "intent": final.get("intent", {}),
        "emotion": final.get("emotion", {}),
        "service_need": final.get("service_need", ""),
        "reply_strategy": final.get("reply_strategy", ""),
        "risk": final.get("risk", {}),
        "order_binding": {
            "binding_status": order_binding.get("binding_status"),
            "target_order_no": order_binding.get("target_order_no"),
            "candidate_order_nos": order_binding.get(
                "candidate_order_nos", []
            ),
            "needs_human_confirmation": order_binding.get(
                "needs_human_confirmation", False
            ),
            "binding_feedback": order_binding.get("binding_feedback", ""),
        },
        "known_facts": known_facts,
        "missing_information": missing_information,
        "recommended_actions": (
            recommended_actions
            if isinstance(recommended_actions, list)
            else []
        ),
    }


def build_prohibited_claims(
    grounding: dict[str, Any],
) -> list[str]:
    """生成回复模型和人工客服都能看到的安全边界。"""

    prohibited = [
        "不得把其他消费者的历史案例事实写入当前回复",
        "不得承诺未经系统确认的退款到账时间、赔偿金额或处理结果",
        "不得把候选订单当成已经确认的目标订单",
        "不得声称已经执行尚未执行的查询、催促、退款或工单操作",
    ]
    missing = grounding.get("missing_information", [])
    if any(
        isinstance(item, dict)
        and item.get("obtain_from") == "business_system"
        for item in missing
    ):
        prohibited.append(
            "系统待查信息只能说将核实，不能编造查询结果"
        )
    return prohibited


def build_template_reply(
    grounding: dict[str, Any],
) -> str:
    """构造不依赖 LLM 的安全兜底回复。

    LLM 服务失败时插件仍应可用，所以模板不是装饰，而是服务可用性的底线。
    """

    emotion = grounding.get("emotion", {})
    emotion_label = str(emotion.get("label") or "")
    if emotion_label in {"不满", "愤怒", "失望", "焦虑"}:
        opening = "很抱歉这次情况给您带来了困扰。"
    else:
        opening = "您好，您的问题我已经了解。"

    binding = grounding.get("order_binding", {})
    binding_status = binding.get("binding_status")
    if binding_status == "ambiguous":
        return (
            opening
            + "为了避免查错订单，麻烦您在订单列表中确认一下本次咨询的商品订单。"
            + "确认后我会继续为您核实处理。"
        )
    if binding_status == "needs_review":
        return (
            opening
            + "当前订单与售后记录的关联需要进一步核对，"
            + "我先为您确认准确记录，再同步处理进展。"
        )

    fact_parts: list[str] = []
    fact_labels = {
        "reship_tracking_no": "补发运单号",
        "reship_carrier": "补发快递公司",
        "original_tracking_no": "原订单运单号",
        "carrier": "原订单快递公司",
        "refund_status": "退款状态",
        "ticket_no": "售后工单号",
    }
    for item in grounding.get("known_facts", []):
        if not isinstance(item, dict):
            continue
        field_code = str(item.get("field_code") or "")
        value = item.get("value")
        if field_code in fact_labels and value not in (None, ""):
            fact_parts.append(f"{fact_labels[field_code]}为 {value}")

    sentences = [opening]
    if fact_parts:
        sentences.append("系统当前记录显示：" + "，".join(fact_parts) + "。")

    customer_missing = [
        item
        for item in grounding.get("missing_information", [])
        if isinstance(item, dict)
        and item.get("obtain_from") == "customer"
    ]
    system_missing = [
        item
        for item in grounding.get("missing_information", [])
        if isinstance(item, dict)
        and item.get("obtain_from") == "business_system"
    ]
    if customer_missing:
        fields = "、".join(
            str(item.get("field") or "相关信息")
            for item in customer_missing
        )
        sentences.append(f"还需要麻烦您补充确认：{fields}。")
    if system_missing:
        fields = "、".join(
            str(item.get("field") or "处理进度")
            for item in system_missing
        )
        sentences.append(
            f"我先为您核实{fields}，确认后第一时间同步给您。"
        )

    if len(sentences) == 1:
        sentences.append("我会根据当前记录继续为您核实处理。")
    return "".join(sentences)


def call_reply_llm(
    grounding: dict[str, Any],
    prohibited_claims: list[str],
    base_url: str,
    model_name: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    """调用 OpenAI 兼容接口，把核验后的事实润色为共情回复。"""

    system_prompt = """
你是美妆电商人工客服的回复辅助插件，只生成供人工审核的回复草稿。

要求：
1. 回复对象是消费者，语气自然、简洁、有共情，不要机械复述标签。
2. 只能使用 known_facts 中已经核验的事实。
3. missing_information 来自 customer 时可以礼貌追问；来自 business_system
   或 human_review 时，只能说明将核实，不得编造结果。
4. order_binding 为 ambiguous 或 needs_review 时，不得自行选择订单。
5. 不得自动发送，不得声称已执行尚未执行的操作。
6. 只返回合法 JSON，不要输出 Markdown。

输出格式：
{
  "reply_text": "",
  "tone": "empathetic|neutral|apologetic",
  "used_fact_codes": [],
  "notes_for_agent": []
}
""".strip()
    user_payload = {
        "grounding": grounding,
        "prohibited_claims": prohibited_claims,
    }
    body = {
        "model": model_name,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        ],
    }
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        response = request_json(url, body, api_key, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 422}:
            raise
        body.pop("response_format", None)
        response = request_json(url, body, api_key, timeout)
    return parse_json_from_text(extract_llm_content(response))


def find_reply_warnings(reply_text: str) -> list[str]:
    """检查草稿中常见的越权承诺，提醒人工客服重点审核。"""

    patterns = {
        r"(保证|一定).{0,8}(到账|解决|赔偿|发出)": "包含确定性结果承诺",
        r"已经为您(退款|赔偿|催促|升级)": "可能声称执行了尚未核验的操作",
        r"百分之百|100%": "包含绝对化表述",
    }
    return [
        warning
        for pattern, warning in patterns.items()
        if re.search(pattern, reply_text)
    ]


def build_reply_result(
    session_id: str,
    grounding: dict[str, Any],
    llm_result: dict[str, Any] | None,
    llm_error: str | None,
) -> dict[str, Any]:
    """合并 LLM 草稿、模板兜底、安全检查和人工审核状态。"""

    template_reply = build_template_reply(grounding)
    llm_reply = ""
    if isinstance(llm_result, dict):
        llm_reply = str(llm_result.get("reply_text") or "").strip()
    reply_text = llm_reply or template_reply

    binding = grounding.get("order_binding", {})
    risk = grounding.get("risk", {})
    missing = grounding.get("missing_information", [])
    review_reasons: list[str] = []
    if binding.get("needs_human_confirmation"):
        review_reasons.append("订单关联需要人工确认")
    if risk.get("level") == "high":
        review_reasons.append("当前会话为高风险服务事件")
    if any(
        isinstance(item, dict)
        and item.get("obtain_from") in {"customer", "human_review"}
        for item in missing
    ):
        review_reasons.append("存在需要消费者或人工补充的信息")
    review_reasons.extend(find_reply_warnings(reply_text))

    return {
        "schema_version": "reply_suggestion_v1",
        "session_id": session_id,
        "delivery_mode": "human_assist_only",
        "suggested_reply": {
            "text": reply_text,
            "status": "draft_for_human_review",
            "generated_by": "llm" if llm_reply else "template_fallback",
            "tone": (
                llm_result.get("tone", "empathetic")
                if isinstance(llm_result, dict)
                else "empathetic"
            ),
        },
        "reply_basis": grounding.get("known_facts", []),
        "missing_information": grounding.get("missing_information", []),
        "agent_actions": grounding.get("recommended_actions", []),
        "prohibited_claims": build_prohibited_claims(grounding),
        "needs_human_confirmation": bool(review_reasons),
        "review_reasons": list(dict.fromkeys(review_reasons)),
        "llm_status": "success" if llm_reply else "fallback",
        "llm_error": llm_error,
        "raw_llm_result": llm_result,
    }


def generate_reply_suggestion(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    analysis: dict[str, Any],
    base_url: str = DEFAULT_LLM_BASE_URL,
    model_name: str = DEFAULT_LLM_MODEL,
    api_key: str = DEFAULT_LLM_API_KEY,
    timeout: int = 120,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """供命令行和未来 Web API 共用的回复生成入口。"""

    session_id = validate_same_session(context, order_binding, analysis)
    grounding = build_grounding_package(order_binding, analysis)
    prohibited_claims = build_prohibited_claims(grounding)
    llm_result: dict[str, Any] | None = None
    llm_error: str | None = None

    if not skip_llm:
        try:
            llm_result = call_reply_llm(
                grounding=grounding,
                prohibited_claims=prohibited_claims,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as exc:  # 回复有模板兜底，因此这里不让全流程崩溃。
            llm_error = str(exc)

    return build_reply_result(
        session_id=session_id,
        grounding=grounding,
        llm_result=llm_result,
        llm_error=llm_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a human-reviewed customer service reply draft."
    )
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--order-binding", type=Path, required=True)
    parser.add_argument("--analysis-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "analysis" / "reply_suggestion.json",
    )
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过回复 LLM，使用可测试的安全模板生成草稿",
    )
    args = parser.parse_args()

    result = generate_reply_suggestion(
        context=read_json(args.context_file),
        order_binding=read_json(args.order_binding),
        analysis=read_json(args.analysis_file),
        base_url=args.llm_base_url,
        model_name=args.llm_model,
        api_key=args.api_key,
        timeout=args.timeout,
        skip_llm=args.skip_llm,
    )
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
