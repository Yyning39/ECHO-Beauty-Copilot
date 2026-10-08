# plan.md — echo-beauty-copilot

## 0. 执行规则（AI 每次任务必须遵守）
- 动手前先复述对本次任务的理解，等我确认
- 一次只推进一个阶段，做完停下等我验收
- 遇到「待确认项」必须停下来问，禁止自行假设
- 发现计划与现实冲突：停下报告冲突点 + 两个方案 + 影响，不要自行修改计划
- 禁止引入未列出的依赖；禁止重构未授权文件
- 新增 Python import 前，先确认 .venv 已安装；未安装的告诉我，不要自动 pip install
- 禁止全局字符串替换。任何「移除/改名」类改动，只改指定位置

## 1. 赛题与最终目标

### 1.1 赛题原文要点
主题：AI 赋能人工客服提质增效，打造「懂消费者、助人工客服、控风险」的
智能服务工具，通过数据实现「服务共情」。

任务细则：
1. 多源数据融合：多模态 AI 解析多源数据，还原消费者情绪、服务轨迹，发现风险
2. AI Agent 构建：用 Agent 技术搭建智能化、自动化的辅助人工客服服务链路

最终产出：
- 产出1 智能接待辅助插件：基于聊天上下文，辅助客服高效定位问题，
  提供精准共情回复
- 产出2 风险异常预警看板：多源数据异常预警 + 闭环状态跟踪

### 1.2 最终目标（对齐赛题）
把 echo-beauty-copilot 做成一个可实际演示、可验证的智能客服辅助系统，
同时覆盖产出1 与产出2，且由同一条 Agent 链路支撑。

- **最终目标 A（产出1）**：客服在工作台打开一个会话，系统基于
  聊天+订单+工单+历史检索，自动给出：问题定位、消费者情绪、共情回复草稿、
  回复依据。人工可一键重新生成、可采纳、可反馈。
- **最终目标 B（产出2）**：系统基于多源数据识别服务异常，形成风险事件，
  支持「识别 → 分派 → 处置 → 已解决」的完整闭环，且过程可追溯。
- **最终目标 C（贯穿）**：Agent 层采用 ReAct 推理范式，让 Agent 自主决定
  调用哪些工具（检索、上下文、订单绑定、风险判定），而非固定顺序。
- **最终目标 D（演示）**：全链路可现场演示——从数据、检索、分析、回复到
  风险闭环，有一条能跑通、能解释每一步依据的路径。

### 1.3 最终目标的验收口径
- 打开会话即看到完整分析（产出1），无需先点「重新生成」
- 能现场创建一条风险事件并走完闭环（产出2）
- Agent 的推理过程可展示、可解释（C）
- 每个结论能说明依据（证据来源、规则、模型输出）
- 有一条固化的演示路径（D）
- 文档口径不夸大：无评估数据就不说「已验证效果」

### 1.4 当前差距（据源码核查）
| 目标 | 当前 | 差距 |
|---|---|---|
| A 产出1 | 60–65% | 首屏未走完整链路；无评估；采纳/反馈现状不明 |
| B 产出2 | 30–40% | 无自动转事件；无状态历史；队列未合并；无闭环记录 |
| C ReAct | 无 | 现为线性链路，Agent 不自主决策 |
| D 演示 | 链路存在但分散 | 无固化演示路径；README 落后 |

## 2. 项目现状（据源码与数据库只读核查）
- 架构：static/app.js → app.py → run_copilot_pipeline.py → 检索/业务上下文
  → order_binding.py → agent_analysis.py → reply_generator.py
- 数据规模：SQLite 中 138 会话 / 998 聊天 / 113 订单 / 80 工单
- 空表：risk_events、agent_runs、reply_feedback 均为 0 条
- risk_events 现有 11 字段：
  id / session_id / buyer_nickname / risk_type / risk_level / reason /
  evidence_json / status / owner / created_at / updated_at
  evidence_json 当前默认保存 app.py 从首屏 evidence 传入的证据文本列表；
  有订单或工单时文本可能包含编号，但不保证每条都有完整来源 ID。
- 状态集：代码允许 待处理/处理中/已解决/已关闭，前端只用前三项
- 无 .git、无 requirements.txt、无 tests/
- 命名：Python snake_case，前端 camelCase
- 首屏数据组装在 app.py:88；「重新生成」调用在 app.js:146
- 关键断点：app.py 无已保存事件时展示 AUTO- 预警，有已保存事件时
  只返回已保存事件 → 预警队列与处置队列未合并

## 3. 本次目标（1 + 2 + ReAct）

### 3.1 产出1 部分
1. 首屏即走完整链路（打开会话即得问题定位/情绪/草稿/依据）
2. 回复草稿可采纳，采纳动作落库
3. 回复可反馈（好/不好 + 可选原因），反馈落库
4. 页面能展示回复依据

### 3.2 产出2 部分
5. 合并预警队列与处置队列（解决 AUTO- 与已保存事件二选一的断点）
6. Agent 分析出的风险可一键转为可跟踪事件（人工确认转，方案A）
7. 打通「识别 → 分派 → 处置 → 已解决」闭环
8. 增加状态变更历史，使闭环过程可追溯
9. 从 database.py:625 移除未使用的「已关闭」

### 3.3 ReAct 部分
10. 新增 ReAct 推理链路，Agent 自主决定工具调用顺序
11. 与线性链路并存，可切换、可对比
12. 推理过程可记录、可展示

### 3.4 演示部分
13. 固化一条可复现的演示路径，同步 README

### 3.5 风险转事件规则（本次已定）
- 转的时机：人工确认转（方案A），不做自动转
- 字段策略：
  - 新增 source_type TEXT，取值 'manual' / 'auto_rule'
  - 不新增 source_rule_id：规则ID写入 reason 前缀 [RULE:xxx]
  - 不新增 source_snapshot：沿用 evidence_json 当前默认传入的证据文本列表；
    不将其表述为完整来源快照或保证包含 ID。
  - 新增 risk_event_history 表：
    event_id / from_status / to_status / operator / reason / created_at
  - 不改 risk_events 现有字段

### 3.6 状态集（已查证）
- 页面与业务统一使用：待处理 / 处理中 / 已解决
- 「已解决」是风险事件闭环的终态；不单设「已关闭」状态。
- 仅从 database.py:625 的 allowed_statuses 移除「已关闭」
- 禁止：改动 app.py:81、cleaning.py:115、cleaning.py:250
- 禁止：全局替换「已关闭」

### 3.7 明确不做
- 不引入 LangChain / LangGraph / 其他 Agent 框架
- ReAct 手写实现，只复用现有 LLM 调用
- 不重构 run_copilot_pipeline.py 为图结构
- 不改 backend/cleaning.py、database.py 的现有表结构（只允许新增字段或新表）
- 不改 scripts/ 下已有脚本的对外函数签名
- 不引入前端框架，保持原生 JS
- 不做图片消息图像理解（列为未来事项）
- 本轮不做模型准确率评估（另开任务）

## 4. ReAct 设计（手写，不引入框架）

### 4.1 目标
在现有线性链路之外，新增 ReAct 推理链路，作为赛题「Agent 技术」的体现。

### 4.2 前置条件（不满足不开始）
- [ ] Git 已初始化，有可回退的 commit
- [ ] requirements.txt 已生成，.venv 可复现
- [ ] 现有线性链路有基线输出（用于对比）
- [ ] 每个工具函数有清晰 docstring

### 4.3 设计原则
- 不替换现有链路，新增并行入口：
  - /api/copilot/analyze → 线性（现有接口，保留，默认）
  - /api/analyze_react  → ReAct（新增）
- 工具直接复用现有函数，不重写
- 必须设最大循环轮数（建议 5）
- 必须有 fallback：轮数用尽或出错时降级到线性链路

### 4.4 工具清单（复用现有）
| 工具 | 来源 | 何时调用 |
|---|---|---|
| retrieve | hybrid_retriever.py | 信息不足需历史证据 |
| get_context | business_context.py | 需还原会话事实 |
| bind_order | order_binding.py | 有订单歧义 |
| assess_risk | agent_analysis.py 风险规则 | 需判定风险等级 |

### 4.5 ReAct 循环
- Thought：LLM 判断信息是否足够
- Action：选工具 + 输入
- Observation：工具返回
- 终止：LLM 输出 Final，或达到最大轮数
- 风险联动：若 assess_risk 返回高风险，触发产出2 的事件建议

### 4.6 可观测性
- 每轮 Thought/Action/Observation 记录
- 在页面展示推理过程（演示亮点）
- 记录到 agent_runs 表（若字段够）

### 4.7 待确认
- LLM 用哪家？（现有 agent_analysis 用什么就沿用）
- 输出格式用 few-shot 还是 JSON？
- 最大轮数设几轮？
- agent_runs 表字段是否够记录每轮？

## 5. 红线（不得越过）
- 【前置】未完成 Git 初始化与 requirements.txt 前，不得开始 Phase 2 之后
- data/echo_beauty.sqlite3 现有数据禁止重导入或覆盖
  - 反例：不得为"重建"跑 import_competition_data.py
  - 任何删除/覆盖 data/ 下文件的操作，先问我要备份方案
- build_vector_store.py 会删除同名 collection，本次禁止运行
- 赛事原始 Excel 不在项目内，禁止修改或迁移
- 新接口返回格式必须与现有 /api/... 一致
- 新增 Python 文件用 snake_case，前端函数用 camelCase
- 修改 app.py:88 首屏数据组装逻辑前，必须先列出受影响字段，等我确认
- 禁止改动 app.py:81、cleaning.py:115、cleaning.py:250
- 禁止全局字符串替换「已关闭」
- 以下文件禁止删除、移动、重命名、改内容：
  - static/cases.json
  - data/analysis/ 下全部 v1/v2/v3 内容

## 6. 演示口径约束
- 可以说："具备创建、记录和更新风险事件的能力"
- 不可以说："已有真实风险闭环数据证明效果"
- 可以说："Agent 采用 ReAct 范式，自主决定工具调用"
- 不可以说："回复质量已验证"（无评估数据）
- risk_events 当前 0 条，演示闭环需现场创建或明示为演示数据
- 任何文档不得编造准确率数字

## 7. 保留素材说明
- data/analysis/ v1/v2/v3：需要保留
- static/cases.json：保留（禁止删除/移动/重命名）

## 8. 每阶段对应赛题的实现

| 阶段 | 对应赛题 | 具体实现 | 对应目标 |
|---|---|---|---|
| Phase 1 | 细则1 多源数据融合 | 盘点首屏 vs 完整链路差异；查采纳/反馈现状；查风险链路断点；查 agent_runs 结构 | 为 A/B 打基础 |
| Phase 2 | 前置工程 | Git + requirements.txt + 基线输出 | 为 C 打基础 |
| Phase 3 | 细则2 AI Agent | 首屏接入完整链路（产出1 主体） | A |
| Phase 4 | 细则2 AI Agent | 手写 ReAct 链路，并行入口 | C |
| Phase 5 | 产出1 交互 | 采纳与反馈落库 | A |
| Phase 6 | 细则1 + 产出2 | 合并预警队列；AUTO- 转事件；移除多余状态 | B |
| Phase 7 | 产出2 闭环 | 状态流转 + risk_event_history | B |
| Phase 8 | 产出1+2 汇总 | 固化演示路径；同步 README | D |

## 9. 分阶段计划

### Phase 1：现状盘点（只读，不改）
- 产出：
  1. 首屏现有字段清单（app.py:88）
  2. 完整链路输出字段清单（run_copilot_pipeline.py）
  3. 两者差异表
  4. 页面「采纳」「反馈」入口现状
  5. reply_feedback 表结构与写入路径
  6. risk_events 风险链路：接口、渲染、断点复现
  7. agent_runs 表结构（能否记录 ReAct 每轮）
  8. 三处「已关闭」确认未被误改
- 验收：1–8 有明确答案；不做任何写操作

### Phase 2：工程前置
- 产出：
  - .gitignore + git init + 首次 commit
  - requirements.txt（扫描现有 import 生成初稿）
  - 现有线性链路的基线输出样本（用于 ReAct 对比）
- 验收：
  - git log 有首次 commit
  - .venv 可据 requirements.txt 复现
  - 基线样本已保存

### Phase 3：首屏接入完整链路（产出1 主体）
- 产出：打开会话即触发完整分析，首屏展示问题定位/情绪/草稿/依据
- 验收：
  - 打开会话无需点「重新生成」即见完整分析
  - 展示字段与完整链路输出一致
  - 有加载态（若响应慢）
  - app.py:81 未被改动

### Phase 4：ReAct 链路（C）
- 产出：
  - /api/analyze_react 入口
  - ReAct 循环 + 工具封装 + 最大轮数 + fallback
  - 推理过程记录到 agent_runs
- 验收：
  - ReAct 能跑通同一会话，输出与线性链路可比
  - 超过轮数时正确降级
  - 推理过程可查询

### Phase 5：采纳与反馈落库（产出1 交互）
- 产出：
  - 页面「采纳」入口 + 落库
  - 页面「反馈」入口 + 落库
  - 回复依据可展开查看
- 验收：
  - 采纳一条草稿，数据库有记录
  - 反馈一条草稿，reply_feedback 有记录
  - 依据展示与后端返回一致

### Phase 6：合并预警队列（产出2 上半）
- 产出：
  - 同一接口同时返回 AUTO- 预警与已保存事件，带 source_type 标记
  - AUTO- 可转为正式事件（人工确认）
  - database.py:625 移除「已关闭」
- 验收：
  - 构造一条 AUTO-，转正后状态可更新
  - source_type 正确落库
  - app.py:81、cleaning.py:115、cleaning.py:250 未被改动

### Phase 7：闭环状态流转 + 历史（产出2 下半）
- 产出：
  - 识别 → 分派 → 处置 → 已解决 的接口与页面操作
  - risk_event_history 表建立，每次状态变更写入
- 验收：
  - 页面完成一次完整流转
  - risk_events 与 risk_event_history 均有记录
  - 历史能回答：谁、何时、从什么状态改到什么

### Phase 8：演示路径与文档（D）
- 产出：
  - 一条可复现的演示步骤（DEMO.md 或写入 README）
  - 更新 README「下一阶段」，与代码现状一致
- 验收：
  - 按步骤跑通
  - README 与代码一致
  - 演示口径符合第 6 节

## 10. 验收方式
- 原有接口返回不变
- 无 tests/，以手动跑通现有流程为准
- Phase 内定义的验收项逐条过
- 演示口径符合第 6 节
- 最终对照第 1.3 节逐条检查

## 11. 偏离处理
- 冲突时停下，输出：冲突点 + 两个方案 + 影响
- 不自行修改本 plan.md，需我确认后由我改

## 附录：术语区分
- 工单状态：业务数据，来自 Excel 清洗，取值如 完成/已完成/已关闭，
  用于 app.py:81 判断会话风险等级。本次不改。
- 风险事件状态：risk_events.status，取值 待处理/处理中/已解决。本次改。
- ReAct：Agent 推理范式（Thought-Action-Observation 循环），
  与前端框架 React 无关。
- 两者同名不同义，禁止混淆。
