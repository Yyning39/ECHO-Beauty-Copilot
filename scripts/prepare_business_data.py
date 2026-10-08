#运行业务数据清洗和关系校验
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.cleaning import load_and_clean_workbook, validate_relations, write_jsonl  # noqa: E402


def main() -> None:
    #1.读取命令行参数；2.加载并清洗excel；3.校验数据关系；4.保存清洗后数据；5.保存质量报告
    parser = argparse.ArgumentParser(description="Prepare cleaned competition business data.")
    parser.add_argument("workbook", type=Path, help="赛事业务数据 Excel 文件路径")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "cleaned",help="清洗结果输出目录")
    args = parser.parse_args()
    #读取和清洗这个工作簿
    data = load_and_clean_workbook(args.workbook)
    #校验主键和跨表关系
    report = validate_relations(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL is convenient here: one business record per line, easy to inspect,
    # stream, diff and import into another database later.
    write_jsonl(args.output_dir / "chat_messages.jsonl", data["chats"])
    write_jsonl(args.output_dir / "orders.jsonl", data["orders"])
    write_jsonl(args.output_dir / "tickets.jsonl", data["tickets"])

    report["source_file"] = str(args.workbook)
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    quality_report_path=(args.output_dir / "quality_report.json")
    quality_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "purpose": "清洗后的赛事业务数据",
        "files": [
            "chat_messages.jsonl",
            "orders.jsonl",
            "tickets.jsonl",
            "quality_report.json",
        ],
        "next_step": "构建服务事件和历史服务记忆 Chunk",
    }

    manifest_path = args.output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
