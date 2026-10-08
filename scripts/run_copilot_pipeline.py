"""运行客服辅助插件的一条完整分析链路。

用户问题
→ BM25 + 向量检索 + RRF
→ 当前业务上下文与历史证据
→ 订单/工单关联
→ 情绪、意图、信息缺口、风险分析
→ 供人工审核的共情回复草稿

这个文件负责“调度”，具体业务规则仍放在各自模块中。这样某一步出错时，
可以单独测试，而不是把所有逻辑堆在一个难以维护的脚本里。
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agent_analysis import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    build_fallback_llm_analysis,
    build_final_result,
    build_llm_input,
    calculate_rule_risk,
    call_llm,
    get_current_messages,
    load_sentiment_pipeline,
    predict_sentiment,
)
from build_bm25_index import load_index, search_bm25
from business_context import build_business_context
from hybrid_retriever import rrf_fuse
from order_binding import build_order_binding, extract_current_case
from query_vector_store import (
    create_query_vector,
    load_manifest,
    search_qdrant,
)
from reply_generator import generate_reply_suggestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "echo_beauty.sqlite3"
DEFAULT_BM25_PATH = PROJECT_ROOT / "data" / "indexes" / "bm25_index.json"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"


def save_json(path: Path, data: dict[str, Any]) -> None:
    """保存阶段产物；default=str 让日期等对象也能安全序列化。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """缓存 Embedding 模型，避免 Web 页面每次分析都重新加载权重。"""

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def get_sentiment_model(model_name: str) -> Any:
    """缓存情感模型；首次加载较慢，后续请求复用同一个模型。"""

    return load_sentiment_pipeline(model_name)


@lru_cache(maxsize=4)
def get_bm25_index(index_path: str) -> tuple[dict[str, Any], Any]:
    """BM25 索引是只读对象，可安全复用，减少磁盘读取和重建开销。"""

    return load_index(Path(index_path))


def run_retrieval_stage(
    session_id: str,
    query: str,
    db_path: Path,
    bm25_path: Path,
    qdrant_path: Path,
    manifest_path: Path | None,
    model_name: str | None,
    collection_name: str | None,
    candidate_k: int,
    top_k: int,
    max_history_cases: int,
    as_of_message_id: str | None,
    as_of_message_seq: int | None,
) -> dict[str, Any]:
    """执行混合检索，并把相似片段还原为可用的业务上下文。"""

    resolved_manifest_path = manifest_path or qdrant_path / "manifest.json"
    manifest = load_manifest(resolved_manifest_path)
    resolved_model_name = model_name or manifest.get(
        "model_name", "BAAI/bge-small-zh-v1.5"
    )
    resolved_collection = collection_name or manifest.get(
        "collection_name", "conversation_memory"
    )

    bm25_index_data, bm25 = get_bm25_index(str(bm25_path.resolve()))
    bm25_results = search_bm25(
        index_data=bm25_index_data,
        bm25=bm25,
        query=query,
        top_k=candidate_k,
    )

    embedding_model = get_embedding_model(str(resolved_model_name))
    current_vector_size = embedding_model.get_embedding_dimension()
    expected_vector_size = manifest.get("vector_size")
    if (
        expected_vector_size is not None
        and int(current_vector_size) != int(expected_vector_size)
    ):
        raise ValueError(
            "Embedding 模型和向量库维度不一致："
            f"{current_vector_size} != {expected_vector_size}"
        )
    query_vector = create_query_vector(embedding_model, query)

    client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists(str(resolved_collection)):
            raise ValueError(f"Qdrant Collection 不存在：{resolved_collection}")
        dense_results = search_qdrant(
            client=client,
            collection_name=str(resolved_collection),
            query_vector=query_vector,
            top_k=candidate_k,
        )
    finally:
        # 本地 Qdrant 会持有文件锁；一次调用结束后主动释放，便于再次运行。
        client.close()

    hybrid_results = rrf_fuse(
        bm25_results=bm25_results,
        dense_results=dense_results,
        top_k=top_k,
    )
    business_context = build_business_context(
        db_path=db_path,
        current_session_id=session_id,
        history_hits=hybrid_results,
        max_history_cases=max_history_cases,
        as_of_message_id=as_of_message_id,
        as_of_message_seq=as_of_message_seq,
    )

    return {
        "query": query,
        "current_session_id": session_id,
        "analysis_cutoff": business_context.get("analysis_cutoff"),
        "retrieval": {
            "embedding_model": resolved_model_name,
            "collection_name": resolved_collection,
            "bm25_result_count": len(bm25_results),
            "dense_result_count": len(dense_results),
            "hybrid_result_count": len(hybrid_results),
            "hybrid_results": hybrid_results,
        },
        "business_context": business_context,
    }


def run_analysis_stage(
    context: dict[str, Any],
    order_binding: dict[str, Any],
    sentiment_model_name: str,
    llm_base_url: str,
    llm_model: str,
    api_key: str,
    timeout: int,
    skip_llm: bool,
) -> dict[str, Any]:
    """执行情感模型、LLM 理解和确定性风险规则。"""

    messages = get_current_messages(context)
    sentiment_model = get_sentiment_model(sentiment_model_name)
    sentiment_results = predict_sentiment(
        sentiment_pipeline=sentiment_model,
        messages=messages,
    )
    llm_input = build_llm_input(
        context=context,
        order_binding=order_binding,
        sentiment_results=sentiment_results,
    )

    llm_status = "skipped" if skip_llm else "success"
    llm_error: str | None = None
    if skip_llm:
        llm_analysis = build_fallback_llm_analysis("本次运行跳过了 LLM")
    else:
        try:
            llm_analysis = call_llm(
                llm_input=llm_input,
                base_url=llm_base_url,
                model_name=llm_model,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as exc:
            llm_status = "failed"
            llm_error = str(exc)
            llm_analysis = build_fallback_llm_analysis("LLM 调用失败")

    rule_risk = calculate_rule_risk(
        messages=messages,
        sentiment_results=sentiment_results,
        llm_analysis=llm_analysis,
    )
    return build_final_result(
        context=context,
        order_binding=order_binding,
        sentiment_model_name=sentiment_model_name,
        sentiment_results=sentiment_results,
        llm_analysis=llm_analysis,
        llm_status=llm_status,
        llm_error=llm_error,
        rule_risk=rule_risk,
    )


def run_copilot_pipeline(
    session_id: str,
    query: str,
    db_path: Path = DEFAULT_DB_PATH,
    bm25_path: Path = DEFAULT_BM25_PATH,
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    manifest_path: Path | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    sentiment_model: str = DEFAULT_SENTIMENT_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str = DEFAULT_LLM_API_KEY,
    timeout: int = 120,
    candidate_k: int = 20,
    top_k: int = 5,
    max_history_cases: int = 3,
    as_of_message_id: str | None = None,
    as_of_message_seq: int | None = None,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """完整流水线的可复用入口，未来 Web API 直接调用这个函数。"""

    timings: dict[str, float] = {}
    total_started = perf_counter()

    started = perf_counter()
    context = run_retrieval_stage(
        session_id=session_id,
        query=query,
        db_path=db_path,
        bm25_path=bm25_path,
        qdrant_path=qdrant_path,
        manifest_path=manifest_path,
        model_name=embedding_model,
        collection_name=collection_name,
        candidate_k=candidate_k,
        top_k=top_k,
        max_history_cases=max_history_cases,
        as_of_message_id=as_of_message_id,
        as_of_message_seq=as_of_message_seq,
    )
    timings["retrieval_context_ms"] = round((perf_counter() - started) * 1000, 2)

    started = perf_counter()
    order_binding = build_order_binding(extract_current_case(context))
    timings["order_binding_ms"] = round((perf_counter() - started) * 1000, 2)

    started = perf_counter()
    analysis = run_analysis_stage(
        context=context,
        order_binding=order_binding,
        sentiment_model_name=sentiment_model,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        api_key=api_key,
        timeout=timeout,
        skip_llm=skip_llm,
    )
    timings["agent_analysis_ms"] = round((perf_counter() - started) * 1000, 2)

    started = perf_counter()
    llm_analysis_succeeded = (
        analysis.get("raw_analysis", {})
        .get("llm", {})
        .get("status") == "success"
    )
    reply = generate_reply_suggestion(
        context=context,
        order_binding=order_binding,
        analysis=analysis,
        base_url=llm_base_url,
        model_name=llm_model,
        api_key=api_key,
        timeout=timeout,
        # 分析 LLM 失败时使用安全模板，避免再基于兜底结果调用一次 LLM。
        skip_llm=(skip_llm or not llm_analysis_succeeded),
    )
    timings["reply_generation_ms"] = round((perf_counter() - started) * 1000, 2)
    timings["total_ms"] = round((perf_counter() - total_started) * 1000, 2)

    return {
        "schema_version": "copilot_pipeline_v1",
        "session_id": session_id,
        "query": query,
        "analysis_cutoff": context.get("analysis_cutoff"),
        "context": context,
        "order_binding": order_binding,
        "analysis": analysis,
        "reply": reply,
        "observability": {
            "stage_timings_ms": timings,
            "human_assist_only": True,
            "auto_send_enabled": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete human customer-service copilot pipeline."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--sentiment-model", default=DEFAULT_SENTIMENT_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-history-cases", type=int, default=3)
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument("--as-of-message-id", default=None)
    cutoff_group.add_argument("--as-of-message-seq", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "analysis" / "copilot_pipeline.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="可选：额外保存 context/binding/analysis/reply 四个阶段文件",
    )
    args = parser.parse_args()

    result = run_copilot_pipeline(
        session_id=args.session_id,
        query=args.query,
        db_path=args.db,
        bm25_path=args.bm25_index,
        qdrant_path=args.qdrant_path,
        manifest_path=args.manifest,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        sentiment_model=args.sentiment_model,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        api_key=args.api_key,
        timeout=args.timeout,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        max_history_cases=args.max_history_cases,
        as_of_message_id=args.as_of_message_id,
        as_of_message_seq=args.as_of_message_seq,
        skip_llm=args.skip_llm,
    )
    save_json(args.output, result)

    if args.artifact_dir is not None:
        save_json(args.artifact_dir / "context.json", result["context"])
        save_json(args.artifact_dir / "order_binding.json", result["order_binding"])
        save_json(args.artifact_dir / "analysis.json", result["analysis"])
        save_json(args.artifact_dir / "reply.json", result["reply"])

    summary = {
        "schema_version": result["schema_version"],
        "session_id": result["session_id"],
        "analysis_cutoff": result["analysis_cutoff"],
        "intent": result["analysis"].get("final", {}).get("intent"),
        "emotion": result["analysis"].get("final", {}).get("emotion"),
        "risk": result["analysis"].get("final", {}).get("risk"),
        "suggested_reply": result["reply"].get("suggested_reply"),
        "observability": result["observability"],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
