#将历史记忆chunk转换为向量，并保存到qdrant
#读取-embedding-创建qdrant collection-将向量和业务源数据写入qdrant-输出


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    读取 JSONL 文件,每一行对应一个历史记忆 Chunk。
    """
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是字典"
                )
            records.append(record)
    return records

def build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    #为每个向量创建payload，即与向量一起保存的业务元数据
    #返回需要知道文本属于哪个对话，哪个用户，关联了什么订单和工单
    return {
        "chunk_id": chunk.get("chunk_id"),
        "parent_id": chunk.get("parent_id"),
        "session_id": chunk.get("session_id"),
        "buyer_nicknames": chunk.get("buyer_nicknames", []),
        "order_nos": chunk.get("order_nos", []),
        "ticket_nos": chunk.get("ticket_nos", []),
        "chunk_index": chunk.get("chunk_index"),
        "message_ids": chunk.get("message_ids", []),
        "start_at": chunk.get("start_at"),
        "end_at": chunk.get("end_at"),

        # 保存原文。
        # 检索命中后，需要把原文交给后续 LLM。
        "text": chunk.get("text", ""),
    }

def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    #创建Qdrant collection，即一组向量数据的集合
    #命名为conversation_memory，专门保存历史服务聊天记录
    #若存在，删除重新创建
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

def save_vectors_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    embeddings: Any,
    batch_size: int,
) -> None:
    #将向量和payload分批写入Qdrant
    total=len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points: list[PointStruct] = []
        for index in range(start, end):
            chunk = chunks[index]
            # 使用数组下标作为 Qdrant 中的点 ID。
            # 当前脚本每次重建索引，因此整数 ID 足够使用。
            point_id = index
            vector = embeddings[index].tolist()
            payload = build_payload(chunk)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        print(f"已写入 {end}/{total} 个向量")

def save_manifest(
    path: Path,
    model_name: str,
    collection_name: str,
    chunk_count: int,
    vector_size: int,
) -> None:
    #保存索引说明文件，以后查询时需要知道：
    #使用了什么模型，向量维度是多少，collection叫什么，写了多少条chunks
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest = {
        "model_name": model_name,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "vector_size": vector_size,
        "distance": "COSINE",
    }
    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector store for historical service memory."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "memory"
            / "memory_chunks.jsonl"
        ),
        help="历史记忆 Chunk 文件",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "qdrant",
        help="Qdrant 本地数据目录",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="conversation_memory",
        help="Qdrant Collection 名称",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding 模型名称",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding 批处理大小",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=64,
        help="Qdrant 批量写入大小",
    )
    args = parser.parse_args()
    # 读取历史记忆 Chunk。
    chunks = read_jsonl(args.input)
    if not chunks:
        raise ValueError("没有读取到任何历史记忆 Chunk")
    # 只对 text 字段进行 Embedding。
    texts = [
        str(chunk.get("text") or "")
        for chunk in chunks
    ]
    if any(not text.strip() for text in texts):
        raise ValueError("存在 text 为空的 Chunk")
    print(f"正在加载 Embedding 模型：{args.model}")
    # 第一次运行时会自动下载模型。
    model = SentenceTransformer(args.model)
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        raise ValueError("无法获取 Embedding 向量维度")
    print(f"向量维度：{vector_size}")
    print(f"正在编码 {len(texts)} 个历史记忆 Chunk")
    # normalize_embeddings=True：
    # 将向量归一化，便于使用余弦相似度。
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # 使用本地持久化模式。
    #
    # 数据会保存在：
    # D:\1ALL\echo-beauty-copilot\data\qdrant
    #
    # 程序结束后数据仍然存在。
    client = QdrantClient(
        path=str(args.qdrant_path),
    )
    # 创建 Collection。
    create_qdrant_collection(
        client=client,
        collection_name=args.collection,
        vector_size=vector_size,
    )
    # 写入向量和 Payload。
    save_vectors_to_qdrant(
        client=client,
        collection_name=args.collection,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=args.upsert_batch_size,
    )
    # 保存构建说明。
    save_manifest(
        path=args.qdrant_path / "manifest.json",
        model_name=args.model,
        collection_name=args.collection,
        chunk_count=len(chunks),
        vector_size=vector_size,
    )
    print("向量库构建完成")
    print(f"Qdrant 路径：{args.qdrant_path}")
    print(f"Collection：{args.collection}")
    print(f"Chunk 数量：{len(chunks)}")


if __name__ == "__main__":
    main()   
