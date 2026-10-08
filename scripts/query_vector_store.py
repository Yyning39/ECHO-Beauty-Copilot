#查询Qdrant的历史服务记忆
#用户问题-加载建库时使用的embedding模型-将当前问题转化为查询向量
#-在Qdrant中搜索相似向量-返回历史chunk和业务源数据

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_manifest(path: Path) -> dict[str, Any]:
    #manifest中保存着建库时的配置，
    #例如：使用的模型，collection名称，向量维度，chunk数量，相似度类型
    if not path.exists():
        return {}
    content = path.read_text(
        encoding="utf-8",
    )
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("manifest.json 的内容不是字典")
    return data
def create_query_vector(
    model: SentenceTransformer,
    query: str,
) -> list[float]:
    #将用户问题转化为查询向量
    if not query.strip():
        raise ValueError("查询问题不能为空")
    embeddings = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    #因为只查询一个问题，所以取下标=0
    query_vector = embeddings[0]
    return query_vector.tolist()

def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    #在Qdrant中进行向量搜索
    #query_vector当前用户问题对应的向量
    #top_k:返回最相似的前几条chunk
    #with_payload=true:要求Qdrant同时返回chunk的业务元数据和原文
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    # query_points() 返回一个查询响应对象。
    # points 中存放具体的命中结果。
    points = response.points
    results: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        results.append(
            {
                # Qdrant 中的点 ID。
                "point_id": point.id,
                # 分数越高，表示向量越相似。
                "score": round(float(point.score), 6),
                # 以下字段来自我们写入 Qdrant 的 Payload。
                "chunk_id": payload.get("chunk_id"),
                "parent_id": payload.get("parent_id"),
                "session_id": payload.get("session_id"),
                "buyer_nicknames": payload.get(
                    "buyer_nicknames",
                    [],
                ),
                "order_nos": payload.get(
                    "order_nos",
                    [],
                ),
                "ticket_nos": payload.get(
                    "ticket_nos",
                    [],
                ),
                "start_at": payload.get("start_at"),
                "end_at": payload.get("end_at"),
                # 最重要的历史聊天原文。
                "text": payload.get("text", ""),
            }
        )
    return results

def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print(f"查询问题：{query}")
    print(f"命中数量：{len(results)}")
    print("=" * 70)

    if not results:
        print("没有找到相关历史记录")
        return

    for index, result in enumerate(results, start=1):
        print(f"结果 {index}")
        print(f"相似度分数：{result['score']}")
        print(f"Chunk ID：{result['chunk_id']}")
        print(f"会话 ID：{result['session_id']}")
        print(f"用户：{result['buyer_nicknames']}")
        print(f"订单号：{result['order_nos']}")
        print(f"工单号：{result['ticket_nos']}")
        print(f"历史内容：{result['text']}")
        print("-" * 70)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query historical service memory from Qdrant."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="需要检索的问题",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "qdrant",
        help="Qdrant 本地数据目录",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="向量库说明文件路径",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant Collection 名称",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding 模型名称",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回最相似的前几条结果",
    )
    args = parser.parse_args()
    # 如果没有手动指定 manifest，
    # 就默认从 Qdrant 数据目录中读取。
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.qdrant_path / "manifest.json"
    manifest = load_manifest(manifest_path)
    # 优先使用命令行传入的配置；
    # 如果没有传入，就使用建库时保存到 manifest 的配置。
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
    print(f"使用模型：{model_name}")
    print(f"使用 Collection：{collection_name}")
    # 如果命令行没有提供问题，
    # 就在终端中交互式输入。
    query = args.query
    if query is None:
        query = input("请输入要检索的问题：").strip()
    if not query:
        raise ValueError("查询问题不能为空")
    # 加载 Embedding 模型。
    model = SentenceTransformer(model_name)
    # 检查模型输出维度是否和建库时一致。
    current_vector_size = (
        model.get_sentence_embedding_dimension()
    )
    expected_vector_size = manifest.get("vector_size")
    if (
        expected_vector_size is not None
        and current_vector_size != expected_vector_size
    ):
        raise ValueError(
            "查询模型的向量维度和建库时不一致："
            f"当前是 {current_vector_size}，"
            f"建库时是 {expected_vector_size}"
        )
    # 将当前问题转换成向量。
    query_vector = create_query_vector(
        model=model,
        query=query,
    )
    # 连接 Qdrant 本地数据库。
    client = QdrantClient(
        path=str(args.qdrant_path),
    )
    # 检查 Collection 是否存在。
    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Collection 不存在：{collection_name}"
        )
    # 执行向量检索。
    results = search_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=args.top_k,
    )
    # 打印结果。
    print_results(
        query=query,
        results=results,
    )

if __name__ == "__main__":
    main()    
