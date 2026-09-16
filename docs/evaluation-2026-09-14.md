# 评测入口与候选任务集

## 当前结论

统一评分与对照运行接口已实现，**真实效果评测尚未完成**。本次没有调用付费或外部模型，也没有把系统自评、历史预测或公开总分当作逐题金标。

数据检查结果：

- 已配置 B 榜问题共 100 题，财报、合同、保险、监管、研报各 20 题；没有逐题标准答案，且比赛开发时已使用，属于历史回放集，不能声称独立留出。
- 历史 `answer_level_v1.jsonl` 有 3 条人工 A 榜开发标注，题目 ID 不在当前 B 榜配置中；其中两题共 5 个原始 chunk 锚点在当前语料中缺失。未导入为可评分标签，也没有拿题号相同或文档相近替代题面版本验证。
- 检查到的 `evaluation_data/official_public/v105_leaderboard_aggregate_v1.json` 是公开总分分布，不是答案文件。
- 新建 12 个自由研究任务草案，覆盖抽取 3、计算 1、比较 2、条款核验 3、短报告 1、证据缺口 1、口径切换 1。它们使用旧语料，尚需人工确认问题可回答性、答案、必要证据及划分；**不是已经批准的独立金标集**。口径切换目前是单次请求，不冒称完成多轮记忆评测。

## 代码入口

- `src/financial_agent/evaluation.py`：EvalCase/EvalSuite，题集版本绑定、独立标签及人工审阅、评分与对照结果检查。
- `src/financial_agent/benchmark.py`：统一的固定 RAG / 动态 Agent 任务适配器，接口不接收标签。
- `scripts/prepare_evaluation.py`：只读旧语料，生成带哈希的题集、待核验标签、文档目录和来源审计。产物写入新的 `data/local/evaluation-*` 目录，避免覆盖旧版本。
- `scripts/run_evaluation_smoke.py`：假模型检验两套适配器与评分器；可选真实语料检索为候选题生成标注材料，网络禁用。
- `scripts/score_evaluation.py`：仅读取保存结果评分，不调用模型或读取密钥。

固定 RAG 采用一次检索和一次合成，不具备动态补查；它是**新建的简单消融对照，不是历史 B 榜第 35 名算法**。动态适配器运行当前研究循环。历史最优方案仍需在其题面/语料/配置被确认后另行接入；本次没有声称复现 92.3949 分。

## 各指标的含义

| 指标 | 实际测量内容 | 不能推出的结论 |
|---|---|---|
| exact_field_accuracy | 严格选项集合、单个数值与单位、明确决策或规范化文本是否符合已批准标签 | 整段分析在语义上正确 |
| citation_integrity_rate | 引用版本、原文哈希、范围与字段是否有效；重复引用不增加有效数量 | 原文真的支持回答中的每个断言 |
| human_answer_accuracy / human_supported_answer_rate / human_completeness_rate | 绑定具体输出哈希的人工复核结果 | 模型自评能替代独立审阅 |
| known_reported_tokens | 服务端明确报告的用量子集 | 未报告部分消耗为零或已知人民币成本 |
| mean_elapsed_seconds | 有记录结果的耗时均值，并报告样本数 | 假模型耗时代表真实模型延迟 |

数值只接受有限十进制字符串并检查单位；0.2 与 20% 不混同。选项不从普通英文解释中抽取 A-H。运行错误不算正确拒答，缺失预测仍在可评分分母中；缺标签的题目不记成答错或答对，准确率为 null。没有引用时引用完整性为 null，不是 100%。

报告/条款等开放答案可使用人工 rubric，不强行做字符串相等；rubric 本身不自动产生分数。人工审阅需要 reviewer、review_kind=human 及精确 prediction_hash，旧输出的审阅不能直接用于新输出。该标记是调用方的来源声明，程序不能证明审阅者身份，需评测流程保证真实性。

禁止在同一题集中混合 fixture、历史回放、候选集和 reviewed_holdout。对照时检查题集、标签、配置和执行模式是否一致；即使数值差值可算，比较函数也不自动授权质量提升结论。真实实验还应核对相同语料/模型/预算、完整样本、重复运行波动和人工盲审。

动态答案仅在 overall 目标完成时输出为 answered；needs_input 映射为 clarification，不自动当作正确拒答。若整体回答恰好引用一个程序计算，导出其值作为结构化字段；不会额外询问模型或从文字中猜选项。字段正确但正文矛盾的情况仍需人工语义审阅。

## 本次已执行验证

- 183 项测试通过：原 148 项 + 新增 35 项。覆盖金标隔离、数字/单位/容差、严格选项解析、缺失预测、错误与拒答区分、引用伪造/重复、人工审阅过期、题集变更、混合集拒绝、对照配置不一致等。
- 两个执行适配器在同一个合成财报 fixture 上跑通。假模型输出中的同比为 20%，固定对照使用 1 次模拟请求，动态循环使用 9 次；**这只是协议测试，不是收益/成本结论，也不能据此判断动态 Agent 更好或更差**。
- 12 个真实任务候选都返回检索候选，共 60 条引用通过原文回取检查；标注材料已落盘。尚未判断检索相关性或完整性，人工批准金标数为 0，真实答案准确率为 null。
- 独立评分命令能重新载入导出的题集、标签、结果和 chunk 文件完成评分；导出保留原始元数据快照，避免旧索引修改元数据造成文档版本漂移。
- Ruff 通过，106 个迁入算法/旧测试文件仍与原始来源哈希一致。

产物路径（相对项目根目录）：

- `data/local/evaluation-20260913T173440Z-915bf57a/`：100 道历史题、12 个任务草案、pending 标签、来源审计。
- `experiments/runs/evaluation-20260913T173907Z-14f66b34/summary.json`：模拟对照与候选材料准备结果。
- 同目录 `annotation_pack.json`：12 题的待核验问题、检索候选全文与空白标注字段。不能把其中 retrieved_candidates_not_gold 直接转为金标。
- `experiments/runs/score-20260913T173950Z-86b57835/report.json`：独立命令重新评分结果。
- `experiments/runs/unit-20260914-evaluation/report.xml`：最终测试记录。

运行目录使用 UTC，本地验证日期为 09-14。所有历史产物保留。本次修改了应用源码，按已有严格恢复策略，旧源码版本的任务数据库不能直接用于本版本续跑；状态查询仍可使用。

## 复现与下一步

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/prepare_evaluation.py
.venv/bin/python scripts/run_evaluation_smoke.py \
  --candidate-suite data/local/evaluation-20260913T173440Z-915bf57a/new_tasks/cases.json
```

通用离线评分命令：

```bash
.venv/bin/python scripts/score_evaluation.py \
  --suite CASES_JSON --labels LABELS_JSON --predictions RUN_JSON \
  --chunks CHUNKS_JSONL
# 有人工逐输出审阅时另加 --reviews REVIEWS_JSON
```

预测包结构：suite_hash、system_id、mode、settings、predictions。每条预测包含 case_id、decision、answer、citations；structured、elapsed_seconds、usage 为可选字段，示例见模拟实验目录。

接下来先为 12 个候选任务核验参考答案和必要证据，按文档/公司/任务族隔离开发集与留出集，避免只换措辞导致泄漏。之后明确真实模型、语料外发范围和预算，再做小规模配对实验。原最高分系统与简化固定 RAG 应作为不同对照分别记录，不应互相替代或混称。记忆、正式报告、交互界面与多轮追问仍是后续实现工作。
