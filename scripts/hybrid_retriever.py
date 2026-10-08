#使用向量检索+bm25                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from build_bm25_index import (
    load_index,
    search_bm25,
)
# 导入向量查询脚本中的函数。
from query_vector_store import (
    SentenceTransformer,
    create_query_vector,
    load_manifest,
    search_qdrant,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def rrf_fuse(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    #bm25_results：BM25 返回的结果列表。
    #dense_results：向量检索返回的结果列表。
    #top_k：最终返回多少条结果。
    #rrf_k：RRF 平滑常数，默认值为 60。
    fused_results: dict[str, dict[str, Any]] = {}
    def add_results(
        results: list[dict[str, Any]],
        source: str,
    ) -> None:
        """
        将某一种检索器的结果加入融合字典。
        source 可能是：
            "bm25"
            "dense"
        """
        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.get("chunk_id")
            # 没有 chunk_id 的结果无法和另一种检索结果合并。
            if not chunk_id:
                continue
            chunk_id = str(chunk_id)
            if chunk_id not in fused_results:
                # 复制结果，避免修改原始查询结果。
                fused_results[chunk_id] = dict(result)
                fused_results[chunk_id][
                    "rrf_score"
                ] = 0.0
                fused_results[chunk_id][
                    "retrieval_sources"
                ] = []
            current = fused_results[chunk_id]
            # RRF 分数只和排名有关。
            current["rrf_score"] += (
                1 / (rrf_k + rank)
            )
            # 保存这个 Chunk 被哪些检索器命中。
            if source not in current[
                "retrieval_sources"
            ]:
                current[
                    "retrieval_sources"
                ].append(source)
            # 保存不同检索器中的排名和原始分数，
            # 方便调试和评估。
            current[f"{source}_rank"] = rank
            if source == "bm25":
                current["bm25_score"] = result.get(
                    "bm25_score",
                    0.0,
                )
            if source == "dense":
                current["dense_score"] = result.get(
                    "score",
                    0.0,
                )
            # 如果当前结果原本没有文本或业务字段，
            # 就使用另一种检索器提供的字段补充。
            #
            # 这样可以保证最终结果尽可能完整。
            for field_name, field_value in result.items():
                if field_name not in current:
                    current[field_name] = field_value
    # 先加入 BM25 结果。
    add_results(
        results=bm25_results,
        source="bm25",
    )
    # 再加入向量结果。
    add_results(
        results=dense_results,
        source="dense",
    )
    # 按 RRF 综合分数从高到低排序。
    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
    # 对分数做格式化，方便展示。
    final_results: list[dict[str, Any]] = []
    for result in ranked_results[:top_k]:
        result["rrf_score"] = round(
            float(result["rrf_score"]),
            6,
        )
        final_results.append(result)
    return final_results

def print_results(
    query: str,
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> None:
    print()
    print(f"查询问题：{query}")
    print("=" * 80)

    print()
    print("BM25 关键词检索结果")
    print("-" * 80)

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"score={result.get('bm25_score')}, "
            f"chunk_id={result.get('chunk_id')}"
        )

    print()
    print("向量语义检索结果")
    print("-" * 80)

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"score={result.get('score')}, "
            f"chunk_id={result.get('chunk_id')}"
        )

    print()
    print("混合检索最终结果")
    print("-" * 80)

    if not hybrid_results:
        print("没有找到相关历史记录")
        return

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        print(f"结果 {rank}")
        print(
            f"RRF 分数："
            f"{result.get('rrf_score')}"
        )
        print(
            f"检索来源："
            f"{result.get('retrieval_sources')}"
        )
        print(
            f"BM25 排名："
            f"{result.get('bm25_rank', '未命中')}"
        )
        print(
            f"向量排名："
            f"{result.get('dense_rank', '未命中')}"
        )
        print(
            f"Chunk ID："
            f"{result.get('chunk_id')}"
        )
        print(
            f"会话 ID："
            f"{result.get('session_id')}"
        )
        print(
            f"用户："
            f"{result.get('buyer_nicknames')}"
        )
        print(
            f"订单号："
            f"{result.get('order_nos')}"
        )
        print(
            f"工单号："
            f"{result.get('ticket_nos')}"
        )
        print(
            f"历史内容："
            f"{result.get('text')}"
        )
        print("-" * 80)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval with BM25 and dense vectors."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="当前用户问题",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "indexes"
            / "bm25_index.json"
        ),
        help="BM25 索引文件",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "qdrant"
        ),
        help="Qdrant 本地数据目录",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="向量库配置文件",
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
        help="每种检索方式先召回多少条候选结果",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="混合检索最终返回多少条结果",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 平滑常数",
    )
    args = parser.parse_args()
    query = args.query
    if query is None:
        query = input(
            "请输入要检索的问题："
        ).strip()
    if not query:
        raise ValueError(
            "查询问题不能为空"
        )
    # 加载 BM25 索引。
    bm25_index_data, bm25 = load_index(
        args.bm25_index
    )
    print("BM25 索引加载成功")
    # 读取 Qdrant 的构建配置。
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = (
            args.qdrant_path
            / "manifest.json"
        )
    manifest = load_manifest(
        manifest_path
    )
    # 优先使用命令行配置；
    # 没有传入时读取 manifest。
    collection_name = (
        args.collection
        or manifest.get(
            "collection_name",
            "conversation_memory",
        )
    )
    model_name = (
        args.model
        or manifest.get(
            "model_name",
            "BAAI/bge-small-zh-v1.5",
        )
    )
    print(
        f"使用 Embedding 模型：{model_name}"
    )
    # 进行 BM25 关键词检索。
    bm25_results = search_bm25(
        index_data=bm25_index_data,
        bm25=bm25,
        query=query,
        top_k=args.candidate_k,
    )
    # 加载和建库时相同的 Embedding 模型。
    model = SentenceTransformer(
        model_name
    )
    # 检查模型维度。
    current_vector_size = (
        model.get_sentence_embedding_dimension()
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
            "查询模型和向量库的维度不一致："
            f"当前模型为 {current_vector_size}，"
            f"向量库为 {expected_vector_size}"
        )
    # 将当前问题转换成查询向量。
    query_vector = create_query_vector(
        model=model,
        query=query,
    )
    # 连接本地 Qdrant。
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

    # 进行向量语义检索。
    dense_results = search_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=args.candidate_k,
    )

    # 使用 RRF 融合两种检索结果。
    hybrid_results = rrf_fuse(
        bm25_results=bm25_results,
        dense_results=dense_results,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )

    print_results(
        query=query,
        bm25_results=bm25_results,
        dense_results=dense_results,
        hybrid_results=hybrid_results,
    )

    # 输出一份 JSON，便于后续 Agent 或接口调用。
    output = {
        "query": query,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "hybrid_results": hybrid_results,
    }

    print()
    print("JSON 结果：")
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
