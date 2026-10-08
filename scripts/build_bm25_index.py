#构建bm25关键词检索索引
#输入：memory_chunks.jsonl
#输出：bm25_index.json
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import jieba
from rank_bm25 import BM25Okapi

PROJECT_ROOT=Path(__file__).resolve().parents[1]

# 用于识别：
# 1. 英文
# 2. 数字
# 3. 订单号
# 4. 工单号
# 5. 连字符和下划线
# 6. 连续中文
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]+"
)

# 判断一个片段是否全部由中文组成。
CHINESE_PATTERN = re.compile(
    r"^[\u4e00-\u9fff]+$"
)
#将常用词组作为整体存入字典
DOMAIN_WORDS = [
    "不良反应",
    "补发换货",
    "线下打款",
    "售后退货",
    "物流工单",
    "物流单号",
    "订单状态",
    "退款金额",
    "退货退款",
    "皮肤泛红",
    "过敏反应",
    "使用感受",
]

for word in DOMAIN_WORDS:
    jieba.add_word(word)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"输入文件不存在：{path}"
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
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

def normalize_text(text: Any) -> str:
    #对文本进行标准化
    normalized = str(text or "").lower()
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )
    return normalized.strip()

def tokenize(text: Any) -> list[str]:
    #分词
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []
    #把normalized_text所有词匹配出来
    segments = TOKEN_PATTERN.findall(
        normalized_text
    )
    tokens: list[str] = []
    for segment in segments:
        if CHINESE_PATTERN.fullmatch(segment):
            # 中文文本使用 jieba 分词。
            words = jieba.lcut(
                segment,
                cut_all=False,
            )
            tokens.extend(words)
        else:
            # 英文、数字、订单号、工单号直接保留。
            tokens.append(segment)

    return [
        token.strip()
        for token in tokens
        if token.strip()
    ]

def build_index_data(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    #tokenized_corpus：所有文档的 Token 列表。用于计算bm25
    #documents：原始 Chunk 和业务元数据。用于返回完整内容
    
    tokenized_corpus: list[list[str]] = []
    documents: list[dict[str, Any]] = []

    for source_index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")

        tokens = tokenize(text)

        # 没有正文的 Chunk 无法参与关键词检索。
        if not tokens:
            continue

        # doc_id 是 BM25 内部使用的编号。
        #
        # 它不一定等于原始文件行号，
        # 所以额外保存 source_index 方便追溯。
        doc_id = len(documents)

        document = dict(chunk)

        document["doc_id"] = doc_id
        document["source_index"] = source_index

        documents.append(document)
        tokenized_corpus.append(tokens)

    if not documents:
        raise ValueError(
            "没有可用于 BM25 检索的有效 Chunk"
        )

    return {
        "algorithm": "BM25Okapi",
        "tokenizer": "jieba",
        "document_count": len(documents),
        "tokenized_corpus": tokenized_corpus,
        "documents": documents,
    }

def save_index(
    path: Path,
    index_data: dict[str, Any],
) -> None:
    #保存 BM25 索引数据。
    #BM25Okapi 对象本身不适合直接保存成 JSON。
    #因此这里保存的是：1. 分词后的语料；2. 原始 Chunk；3. 索引配置
    #以后读取时，再根据 tokenized_corpus，重新创建 BM25Okapi 对象。
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = dict(index_data)
    output["built_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_index(
    path: Path,
) -> tuple[dict[str, Any], BM25Okapi]:
    """
    加载已经保存的 BM25 索引。
    返回两个对象：
        index_data：
            保存的 Chunk、Token 和配置。
        bm25：
            重新创建的 BM25 检索对象。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 索引不存在：{path}"
        )
    index_data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    #tokenized_corpus文档的token列表
    tokenized_corpus = index_data.get(
        "tokenized_corpus"
    )
    if not tokenized_corpus:
        raise ValueError(
            "BM25 索引中没有 tokenized_corpus"
        )
    bm25 = BM25Okapi(
        tokenized_corpus
    )
    return index_data, bm25

def search_bm25(
    index_data: dict[str, Any],
    bm25: BM25Okapi,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    使用 BM25 查询历史 Chunk。
    查询步骤：
        1. 将用户问题分词
        2. 计算每个文档的 BM25 分数
        3. 按分数从高到低排序
        4. 返回前 top_k 个 Chunk
    score 越高，表示关键词相关性越强。
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    documents = index_data["documents"]
    scores = bm25.get_scores(
        query_tokens
    )
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[index])
        # 分数为 0 表示没有匹配到有效关键词。直接返回不提供给用户
        if score <= 0:
            continue
        result = dict(documents[index])
        result["bm25_score"] = round(
            score,
            6,
        )
        # 记录查询中实际出现的 Token，
        # 方便调试 BM25 为什么命中。
        document_tokens = set(
            index_data["tokenized_corpus"][index]
        )
        result["matched_query_tokens"] = [
            token
            for token in query_tokens
            if token in document_tokens
        ]
        results.append(result)
        if len(results) >= top_k:
            break
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
        print("没有找到关键词匹配的历史 Chunk")
        return
    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"结果 {rank}")
        print(f"BM25 分数：{result['bm25_score']}")
        print(f"Chunk ID：{result.get('chunk_id')}")
        print(f"会话 ID：{result.get('session_id')}")
        print(f"用户：{result.get('buyer_nicknames')}")
        print(f"订单号：{result.get('order_nos')}")
        print(f"工单号：{result.get('ticket_nos')}")
        print(
            f"匹配词：{result.get('matched_query_tokens')}"
        )
        print(f"历史内容：{result.get('text')}")
        print("-" * 70)


def main() -> None:
    """
    程序入口。
    这个脚本支持两种模式：
        1. 只构建索引：
        2. 构建索引后立即测试查询：
    """
    parser = argparse.ArgumentParser(
        description="Build and test BM25 index."
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
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "indexes"
            / "bm25_index.json"
        ),
        help="BM25 索引输出路径",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="可选：构建完成后立即测试的问题",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最多返回多少条结果",
    )
    args = parser.parse_args()
    # 读取历史 Chunk。
    chunks = read_jsonl(args.input)
    # 构建分词数据和原始文档数据。
    index_data = build_index_data(chunks)
    # 保存索引数据。
    save_index(
        path=args.output,
        index_data=index_data,
    )
    # 重新加载索引并创建 BM25 对象。
    # 这样可以验证：
    # 保存后的文件能够被正常恢复。
    loaded_data, bm25 = load_index(
        args.output
    )
    print("BM25 索引构建完成")
    print(f"索引文件：{args.output}")
    print(
        f"文档数量：{loaded_data['document_count']}"
    )
    print(
        f"词表 Token 数量："
        f"{len(set(token for tokens in loaded_data['tokenized_corpus'] for token in tokens))}"
    )
    # 如果命令行传入 query，
    # 则立即执行一次查询测试。
    if args.query:
        results = search_bm25(
            index_data=loaded_data,
            bm25=bm25,
            query=args.query,
            top_k=args.top_k,
        )
        print_results(
            query=args.query,
            results=results,
        )


if __name__ == "__main__":
    main()