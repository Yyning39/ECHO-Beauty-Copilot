# ECHO Beauty

美妆服务共情副驾：面向人工客服的服务轨迹理解、回复辅助和风险闭环原型。

## 现在包含

- 客服工作台右侧 AI 辅助插件原型
- 赛事 Excel 清洗、关系校验和 SQLite 业务事实库
- BM25 + BGE 向量检索 + RRF 混合排序的历史服务记忆
- 当前会话时间截断，避免离线评估读取未来消息和未来工单
- 订单/工单关联核验、Erlangshen 情感分析、LLM 语义分析和风险规则
- 用户订单、会话和工单的服务轨迹视图
- 固定业务意图结构、情绪趋势、信息缺口和风险等级展示
- 带核验依据的共情回复草稿（仅供人工审核，不自动发送）
- 风险事件创建、升级主管和状态跟踪
- 独立的风险中心看板

工作台中的“重新生成”按钮已经接入真实 Agent 流水线，不再使用固定回复。

## 数据库

数据库文件位于 `data/echo_beauty.sqlite3`，当前包含：

- `buyers`：用户；
- `sessions`：会话；
- `chat_messages`：聊天记录；
- `orders`：订单；
- `tickets`：统一后的售后工单；
- `service_events`：用户服务时间线；
- `risk_events`：风险事件；
- `agent_runs`：Agent 执行日志。

重新导入赛事数据：

```powershell
python scripts/import_competition_data.py "D:\Program Files (x86)\download\赛题 1：数据共情者-业务数据.xlsx"
```

导入脚本会保留原始字段 JSON，并重建服务时间线，便于后续校验和开发查询接口。

## 已接入接口

服务启动后提供：

- `GET /api/health`：检查服务和数据库；
- `GET /api/cases?limit=20`：返回工作台会话列表；
- `GET /api/cases/{session_id}`：返回单个会话的聊天、订单、工单和时间线；
- `GET /api/risk-events`：返回规则预判和已保存的风险事件；
- `POST /api/risk-events`：保存人工确认创建的风险事件；
- `PATCH /api/risk-events/{id}`：把风险事件更新为待处理、处理中、已解决或已关闭；
- `POST /api/copilot/analyze`：运行检索、关联、分析和回复草稿生成；
- `POST /api/reply-feedback`：记录人工采纳、编辑或拒绝 AI 草稿；
- `GET /api/stats`：返回数据规模统计。

`POST /api/copilot/analyze` 的请求示例：

```json
{
  "session_id": "S00001",
  "query": "收到的粉底液泵头坏了，怎么办？",
  "as_of_message_id": "20974795108539.PNM",
  "skip_llm": false
}
```

`as_of_message_id` 表示只分析该消息及之前的数据，主要用于离线回放和评估。

## 数据清洗

第一阶段先将赛事 Excel 清洗成统一业务记录，并生成关系质量报告：

```powershell
python scripts/prepare_business_data.py "D:\Program Files (x86)\download\赛题 1：数据共情者-业务数据.xlsx"
```

输出位于 `data/cleaned/`：

- `chat_messages.jsonl`：标准化聊天记录；
- `orders.jsonl`：标准化订单；
- `tickets.jsonl`：统一格式的售后工单；
- `quality_report.json`：重复数据和跨表关联检查结果。

清洗阶段不调用大模型，也不删除原始 Excel。原始数据作为事实来源，清洗结果供后续服务时间线、历史记忆和 RAG 使用。

## 启动

在本目录执行：

```powershell
python app.py
```

然后打开 http://127.0.0.1:8000 。

如果使用 DeepSeek 等 OpenAI 兼容接口，先在当前 PowerShell 设置：

```powershell
$env:OPENAI_API_BASE = "https://api.deepseek.com/v1"
$env:LLM_MODEL_NAME = "你的模型名称"
$env:DEEPSEEK_API_KEY = "你的密钥"
```

也可以不启动页面，直接验证完整链路：

```powershell
.\.venv\Scripts\python.exe scripts\run_copilot_pipeline.py `
  --session-id S00001 `
  --query "收到的粉底液泵头坏了，怎么办？" `
  --as-of-message-id "20974795108539.PNM" `
  --output data\analysis\copilot_pipeline.json
```

使用 `--skip-llm` 时仍会运行检索、订单关联、情感模型和风险规则，回复由安全模板兜底。

## 下一阶段

1. 增加回复“采纳、编辑、拒绝”的人工反馈记录。
2. 完成风险事件的分派、处理中、已解决闭环状态更新。
3. 为图片消息接入破损/不良反应等多模态分析。
4. 建立意图、情绪、订单关联、风险和回复质量评估集。
5. 保存 LLM token 用量、模型调用状态和命中证据，形成成本与质量看板。

## 项目定位

核心闭环：

```text
聊天消息 → 轨迹融合 → 意图/情绪/风险识别 → 知识检索 → 回复建议 → 人工确认 → 风险跟踪
```
