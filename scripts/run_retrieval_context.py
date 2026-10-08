#输入：当前会话id和用户问题
#处理：bm25关键词检索、Qdrant向量检索、rrf混合排序，sqlite查询当前会话事实，sqlite查询历史会话事实、生成最终业务上下文
#输出：可以交给llm的结构化上下文
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
# 导入 BM25 索引和查询方法。
from build_bm25_index import (
    load_index,
    search_bm25,
)
# 导入向量库查询方法。
from query_vector_store import (
    create_query_vector,
    load_manifest,
    search_qdrant,
)
# 导入混合检索中的 RRF 融合方法。
from hybrid_retriever import rrf_fuse
# 导入 SQLite 业务上下文构建方法。
from business_context import (
    build_business_context,
    render_context_text,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval and build business context."
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="当前正在处理的会话 ID",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="当前用户问题",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "echo_beauty.sqlite3",
        help="SQLite 数据库路径",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "indexes"
        / "bm25_index.json",
        help="BM25 索引路径",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "qdrant",
        help="Qdrant 本地数据路径",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="向量库 manifest 文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding 模型名称",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant Collection 名称",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="每种检索方法召回的候选数量",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="混合检索最终返回数量",
    )
    parser.add_argument(
        "--max-history-cases",
        type=int,
        default=3,
        help="最多补充多少个历史服务会话",
    )
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--as-of-message-id",
        type=str,
        default=None,
        help="只分析该消息及之前的数据，防止离线测试读取未来消息",
    )
    cutoff_group.add_argument(
        "--as-of-message-seq",
        type=int,
        default=None,
        help="按消息序号设置离线回放截断点",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出文本或 JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：将最终上下文保存为 JSON 文件",
    )
    args = parser.parse_args()
    # 读取向量库构建时保存的配置。
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = (
            args.qdrant_path
            / "manifest.json"
        )
    manifest = load_manifest(
        manifest_path
    )
    model_name = (
        args.model
        or manifest.get(
            "model_name",
            "BAAI/bge-small-zh-v1.5",
        )
    )
    collection_name = (
        args.collection
        or manifest.get(
            "collection_name",
            "conversation_memory",
        )
    )
    print(
        f"使用 Embedding 模型：{model_name}"
    )
    print(
        f"使用 Qdrant Collection："
        f"{collection_name}"
    )

    # 第一步：加载 BM25 索引。
    bm25_index_data, bm25 = load_index(
        args.bm25_index
    )

    # 第二步：执行 BM25 关键词检索。
    bm25_results = search_bm25(
        index_data=bm25_index_data,
        bm25=bm25,
        query=args.query,
        top_k=args.candidate_k,
    )

    print(
        f"BM25 命中："
        f"{len(bm25_results)} 条"
    )

    # 第三步：加载和建库时相同的 Embedding 模型。
    model = SentenceTransformer(
        model_name
    )

    # 检查向量维度。
    #
    # 查询模型的输出维度，
    # 必须和 Qdrant Collection 的维度一致。
    current_vector_size = (
        model.get_embedding_dimension()
    )

    expected_vector_size = manifest.get(
        "vector_size"
    )

    if (
        expected_vector_size is not None
        and current_vector_size
        != expected_vector_size
    ):
        raise ValueError(
            "查询模型和向量库维度不一致："
            f"当前模型为 {current_vector_size}，"
            f"向量库为 {expected_vector_size}"
        )

    # 第四步：将当前问题转换为向量。
    query_vector = create_query_vector(
        model=model,
        query=args.query,
    )

    # 第五步：连接本地 Qdrant。
    client = QdrantClient(
        path=str(args.qdrant_path)
    )

    if not client.collection_exists(
        collection_name
    ):
        raise ValueError(
            f"Qdrant Collection 不存在："
            f"{collection_name}"
        )

    # 第六步：执行向量检索。
    dense_results = search_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=args.candidate_k,
    )

    print(
        f"向量检索命中："
        f"{len(dense_results)} 条"
    )

    # 第七步：使用 RRF 融合两种检索结果。
    hybrid_results = rrf_fuse(
        bm25_results=bm25_results,
        dense_results=dense_results,
        top_k=args.top_k,
    )

    print(
        f"混合检索结果："
        f"{len(hybrid_results)} 条"
    )

    # 第八步：根据当前 session_id，
    # 查询当前会话的完整业务事实；
    # 同时根据混合检索结果查询历史业务事实。
    business_context = build_business_context(
        db_path=args.db,
        current_session_id=args.session_id,
        history_hits=hybrid_results,
        max_history_cases=args.max_history_cases,
        as_of_message_id=args.as_of_message_id,
        as_of_message_seq=args.as_of_message_seq,
    )

    # 组织最终输出。
    #
    # 这个结构后续可以直接传给：
    #
    #     意图分析模块
    #     情绪分析模块
    #     风险判断模块
    #     LLM 回复模块
    final_context = {
        "query": args.query,
        "current_session_id": args.session_id,
        "analysis_cutoff": business_context.get(
            "analysis_cutoff"
        ),
        "retrieval": {
            "bm25_result_count": len(
                bm25_results
            ),
            "dense_result_count": len(
                dense_results
            ),
            "hybrid_result_count": len(
                hybrid_results
            ),
            "hybrid_results": hybrid_results,
        },
        "business_context": business_context,
    }

    # 如果指定了 output，
    # 就将最终结果保存成 JSON。
    if args.output is not None:
        save_json(
            path=args.output,
            data=final_context,
        )

        print(
            f"最终上下文已保存："
            f"{args.output}"
        )

    # 根据参数选择终端输出格式。
    if args.format == "json":
        print(
            json.dumps(
                final_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print(
            render_context_text(
                business_context
            )
        )


if __name__ == "__main__":
    main()
