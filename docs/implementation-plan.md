# 金融长文档 Agent：基于现有实现的增量改造设计

日期：2026-09-13。本文件保留设计时点的计划；2026-09-14 已开始批次 A 增量实现，当前完成度以项目 README 和验证记录为准，不表示本文所有能力已实现。

## 1. 本次决策

以现有金融长文档问答系统为技术主体，增加统一研究任务层，形成支持信息提取、财务计算、条款核验、跨文档比较、综合报告及连续追问的 Agent。保留多领域能力，不缩成单一公司的财报 Demo，不以行情预测或自动交易为目标。

“完整”的工程标准：可接收新材料和自由任务；能依据结果选择下一步工具；能回取证据、执行可靠计算并报告缺口；能在限定预算内结束；能保存任务并恢复；有多类型测试与可展示入口。不是要求建设商业化金融平台。

实施策略：保留比赛代码和结果快照；在开发副本/开发分支中增加 `financial_agent/` 应用层。旧模块先通过适配器复用，不进行大规模目录改名。实现与比赛导出隔离；新项目核心不得读取历史答案、榜单标签或按题号查答案。

## 2. 本轮代码核对结果

发布目录：`/home/m1kasa/repo/afac2026-financial-longtext-agent-team`，HEAD `b8b6c20271bfc3b452af25474094625d456322e1`。

历史目录：`/home/m1kasa/repo/AFAC赛题四_团队方案_v1_20260712`，HEAD `7ae876fd5b5c2d617e6c90ddd6374c87b5180e96`。历史目录有用户既有改动，因此本次比较的是磁盘文件，不能全部归因于该 commit。

已执行：

- AST 检查发布目录 `agent`、`agent_team_*` 中 144 个 Python 文件，无语法解析错误；不是整个仓库的运行测试。
- 与历史目录逐文件比较：94 个字节一致，1 个不同，49 个仅发布目录存在。唯一不同的共同文件为 `agent/llm/qwen_client.py`，差异是 HTTP 错误的抛出方式。此统计不表示 94/144 的项目完成度或贡献占比。
- 检出 2 个直接缺失的本地模块，两者均在历史目录找到；检查其静态本地导入，候选恢复后未发现额外缺失。动态导入、资源路径和第三方依赖不在该结论范围内。
- 对当前发布代码直接运行历史目录的 4 个事实账本测试、6 个计算 DSL 测试，10/10 通过。使用轻量测试运行器而非 pytest；仅覆盖这些无 fixture 的纯函数测试。
- 未读取密钥内容，未调用模型，未运行付费评测，未修改两个仓库中的业务文件。

待恢复模块与校验值：

| 模块 | 历史源码 SHA-256 |
| --- | --- |
| `agent_team_v7/adaptive_blind.py` | `abcfa3a7cd049fae49631a459e6e1f838ab71e2feb20d71214cac78536008ab6` |
| `agent_team_v9/option_coverage_index.py` | `04b3f0fc22868463ce469a602140574e6a436ea8584a138048672f4130c71d8b` |

## 3. 保留、整合、增强、新增

以下路径均相对发布仓库；同名目标模块为拟议新增文件，而非已存在实现。

| 类别 | 现有模块 | 改造动作 | 拟议接入位置 |
| --- | --- | --- | --- |
| 保留 | `agent/preprocess/layout_pdf.py`、金融表格解析 | 包装文档接入服务，保存解析版本、物理页与原文位置，不重写解析算法 | `financial_agent/documents/service.py` |
| 恢复 | `agent_team_b1/retrieval_v1.py` 的两个历史依赖 | 从已定位的旧源恢复，补来源清单和回归测试 | 原 import 路径兼容保留 |
| 保留并整合 | `agent/index/bm25.py`、`document_index.py`、`agent/retrieve/` | 对外暴露统一 SearchRequest，不要求自由任务伪装成选择题 | `adapters/retrieval.py` |
| 泛化 | `domain_option_target_v21.py` | 保留分目标检索思路；选项仅是 Target 的一种，不是所有研究任务都生成 A/B/C/D | `tools/search_evidence.py` |
| 整合 | `agent/reasoning/logicrag.py`、`multi_logicrag.py` | 复用子问题、依赖校验、充分性与改写函数；把每次可继续步骤暴露给运行时 | `orchestration/planner.py`、`adapters/planning.py` |
| 增强 | `focused_reasoning_v11.py`、`agent/compress/rule_filter.py` | 保留聚焦摘录，新增预算触发、来源范围和关键事实保留检查 | `memory/context_builder.py` |
| 增强 | `fact_ledger.py` | 在原事实字段之外补实体、币种、期间类型、来源版本和状态；用适配器保持旧数据兼容 | `evidence/facts.py` |
| 增强 | `calculation_dsl.py` | 保留 Decimal 和白名单运算；入口补完整口径检查及派生值来源链 | `tools/calculate.py` |
| 整合 | `claim_verifier.py`、`claim_set_verifier.py`、`document_bound_v12.py` | 将选项判断推广为报告 Claim；分开“引用存在”与“证据支持” | `verification/service.py` |
| 保留为兼容模式 | `agent/reasoning/solver.py`、B1 solver | 原题目问答可继续使用；不直接把旧 Solver 当所有自由任务的唯一入口 | `adapters/competition.py` |
| 归档/回归 | `financial_unrounded_v45.py`、`submission_v45.py` | 保留原提交重建及精度回归；通用 Agent 不调用按题号替换答案逻辑 | `legacy` 逻辑边界，不强制移动文件 |
| 新增 | 无统一自由研究任务接口 | 会话、工具协议、任务状态、记忆存储、报告、接口和界面 | `financial_agent/` |

## 4. 需要优先解决的真实接口问题

### 4.1 自由任务不能直接套比赛 Question

`agent/schemas.py` 的 AnswerFormat 只有 mcq/multi/tf；BQuestion 的 freeform 仍带固定答案槽位。`B1Retriever.adapt_question()` 会把 freeform 映射为 mcq。这是旧检索复用手段，不适合作为新应用的领域模型。

方案：增加独立 TaskRequest、ResearchTarget、ResearchResult。比赛 Question/BQuestion 只在兼容适配器边界使用；新工具接口不暴露 qid、split 或官方答案模板。

### 4.2 文档范围必须由执行层保证

`B1Retriever.retrieve()` 忽略传入 restrict_to_doc_ids，适配时清空 doc_ids，符合其盲检用途，但不能直接承担用户指定材料范围的隔离。

方案：SearchRequest 带 allowed_document_versions；执行层将模型要求的范围与授权范围取交集，在检索前过滤，在结果返回后再核验。不得把范围控制只写进提示词。文档 domain 用于检索提示，不是权限边界。

### 4.3 现有短摘要不等于持久化记忆

LogicRAG 的 rank 记忆包含 80 字截取；另一摘要路径也会按字符截断。可保留为便宜的工作摘要，但事实、来源和未解决冲突需要独立保存。检查点、对话摘要、证据存储应分开建模。

### 4.4 实际用量与估算不能混写

现有 QwenClient 在 usage 缺失时按字符估算，返回相同 TokenUsage 结构；内部 HTTP 重试也不能完整体现为外部工具步骤。

方案：新应用 transport 单独保存每次 attempt 的 raw_usage、usage_status（reported/estimated/unknown）、估算值与错误。预算可利用估算保守预留，但审计不能把它称为实际消耗。统一在 transport 实施重试，不同时叠加旧客户端与新运行时的多层重试。记录工具可观察的动作和证据，不依赖保存模型内部思考文本。

### 4.5 计算入口需要强约束

现有候选计算生成有部分口径筛选；但直接调用 evaluate_calculation 并不等于全面执行币种、主体、期间和报表口径校验。另有增长公式使用绝对值基期，应明确命名、标注公式政策，负基期时不能默认为所有业务认可的“同比”。

方案：在新工具入口校验单位与口径，公式版本化；遇到不明确的负基期、重述或币种转换需求时返回需确认，而不是自动套公式。

## 5. 总体结构

```text
CLI / Web / API
       │
TaskService：任务、会话、范围、取消
       │
Agent Runtime：规划 → 选择工具 → 执行 → 更新状态 → 核验/补查
       │                   │
       │                   └─ 执行预算、幂等键、错误与检查点
       │
工具注册表：定位文档 / 检索 / 回读 / 提取事实 / 计算 / 核验
       │
适配层：复用现有解析、BM25、LogicRAG、压缩、事实账本、DSL
       │
文档库 + EvidenceStore + FactStore + TaskStore
       │
ReportRenderer：答案、对比表、计算来源、证据链接、未解决项
```

采用单主控 Agent；核验是可独立调用的能力，不必为了名称引入多个会话互相辩论。固定的是安全状态机，不是具体研究步骤。简单问题可直接检索作答，复杂问题才展开多步任务；路由结果与原因进入事件记录。

拟议目录：

```text
financial_agent/
  contracts/          # 自由任务、证据、事实、工具、结果
  adapters/           # 旧实现的窄接口，不搬入历史答案
  documents/          # 文件版本、解析与索引清单
  orchestration/      # 主控、状态机、计划修订、预算
  tools/              # 可调用能力与注册表
  evidence/           # 引用绑定、事实与计算来源
  memory/             # 工作上下文压缩、跨轮次更新
  storage/            # SQLite 检查点与事件
  verification/       # 结论核验与缺口归类
  reports/            # 研究结果渲染
  api/                # 应用接口
tests/financial_agent/
  unit/ integration/ acceptance/
```

编排框架可采用前轮已调研的 LangGraph，但 contracts/tools/domain 模块不依赖其专有类型，避免把框架替换等同于算法重写。SQLite 用于本地单用户持久化；第一版不引入 Redis/分布式队列。库版本在实施时锁定并验证，本文不把框架选型当已安装事实。

## 6. 最小接口契约

以下为设计，不是已经可调用的 API。

| 对象 | 必要字段 | 核心规则 |
| --- | --- | --- |
| TaskRequest | task_id、session_id、query、document_version_ids、output_kind、budget | output_kind 为 answer/comparison/report；用户不需要传题型或选项 |
| ResearchTarget | target_id、question、depends_on、required_facets、status | DAG 无环；可追加子问题；用实体、期间、条款例外等 facet 判断缺口 |
| EvidenceRef | evidence_id、doc_id、version_id、physical_page、chunk_id、source_span、text_hash | 每条引用可精确回取；展示页码与 PDF 物理页区别存储 |
| FinancialFact | fact_id、entity、metric、period、period_kind、value、unit、currency、scope、restatement、evidence_ids、status | 数字用十进制字符串；未知字段不得由默认值伪装成已确定 |
| CalculationResult | calculation_id、formula_id、formula_version、operand_ids、unrounded_value、display_value、unit | 所有输入有来源；派生运算可引用已验证计算结果，禁止循环 |
| Claim | claim_id、text、evidence_ids、calculation_ids、status、limitations | status 为 supported/contradicted/insufficient；模型自信不是已支持 |
| ToolResult | call_id、status、payload_refs、missing_facets、usage_record_ids、error | status 为 ok/partial/needs_input/error；不以任意超长文本充当全部状态 |
| ResearchResult | task_id、answer、claims、tables、unresolved、sources、run_summary | 局部成功与完整成功明确区分；有结论不代表所有目标已完成 |

服务端生成不可猜测 ID；哈希用于完整性而不是权限。工具参数中的事实 ID、文档 ID 必须属于当前任务可访问集合。工具不得自行扩大文件系统读取范围。

工具契约：

| 工具 | 输入 | 输出与拒绝条件 |
| --- | --- | --- |
| find_documents | 主体/期间/类型、允许范围 | 候选版本与缺失项；不能访问范围外材料 |
| search_evidence | target、queries、允许范围、检索预算 | EvidenceRef 列表和检索覆盖信息；空结果不等于原文不存在 |
| read_source | evidence_id 或 doc_version/page/span | 原文与上下文；拒绝失效 ID、篡改哈希、越界范围 |
| extract_facts | target、evidence_ids | 候选事实及口径缺口；不从模型常识补金额 |
| calculate | formula_id、operand_ids、display_precision | 计算来源链；口径冲突/零分母/未知公式返回明确错误 |
| verify_claims | claims、引用集合 | 逐项结论状态及缺失 facet；需要时触发补查 |

计划选择、压缩、结束和询问属于运行时动作，不开放任意脚本或任意网络访问。报告渲染是输出服务，不需要设计为一个能随意写文件的模型工具。

## 7. 状态、记忆与预算

任务状态：created → running → completed / partial / needs_input / failed / cancelled。needs_input 收到补充后回到 running；临时模型/网络故障的恢复由检查点记录决定，不把 failed 默默改成成功。completed 后的追问创建新 task_id，沿用 session_id 与允许复用的事实。

工作循环：选择尚未解决 target → 选择工具 → 校验权限与预算 → 执行并落盘 → 合并证据/事实 → 判断下一步。计划修订只改变未完成目标，保留旧版计划和变更理由。

三层存储：原始证据不可变；事实账本可增加新版本及冲突关系；工作记忆可压缩。追问换主体/期间时重算依赖，不能仅编辑摘要。文档被新版本替换后，旧结论保留历史版本标签，当前结果不得悄悄混用。

初始建议参数（可配置、待实测）：最多 12 次工具步骤，每目标最多 2 次补查，相同参数无新证据的重复调用停止；另设共享 LLM 请求和 Token 预算。不得通过把多个内部调用包装成一次工具来绕过限制。预算覆盖规划、摘要、核验、修复及重试。预留生成最终结果的额度。

检查点原子保存状态版本、事件序号、工具完成记录和结果引用。执行层用 expected_state_version 防止同任务并发覆盖。已保存的成功工具结果恢复时不重新执行；外部请求已发出而响应丢失时标记 uncertain，不宣称 API 恰好执行一次。停止/取消时保留部分结果和未完成原因。

## 8. 增量实施批次

详细依赖和验收用例见同目录 `backlog.json`。

### 批次 A：兼容底座与边界

- FA-01 恢复依赖与旧测试，记录来源和版本。
- FA-02 创建自由任务/证据/结果契约，隔离比赛适配器。
- FA-03 文档版本及范围受控检索，复用已有解析索引。
- FA-04 模型 transport 与 usage/重试账务。

退出条件：依赖可安装、纯离线测试通过；不依赖答案快照；指定文档范围不能泄漏；模型关闭时不会意外发请求。

### 批次 B：Agent 主链路

- FA-05 事实与计算工具，补运行入口口径校验。
- FA-06 任务状态/检查点/事件存储。
- FA-07 复用 LogicRAG 的动态调度和补查，统一预算。
- FA-08 报告 Claim 核验和结果渲染。

退出条件：抽取、计算、条款核验、跨文档比较都能通过同一个任务接口执行；中途证据不足能补查或结束；不强制每题多轮规划。

### 批次 C：连续研究与交付

- FA-09 三层记忆、压缩和追问失效处理。
- FA-10 API 与最小界面：上传、任务、引用定位、继续追问、取消。
- FA-11 多领域评测、消融与故障测试。
- FA-12 环境锁定、运行文档、演示与最终项目说明。

退出条件：所有确定性验收检查通过，真实模型质量与成本有独立报告，用户可替换文档和提问而不修改源码。最终描述使用实测结果，不把规划中的指标写进简历。

## 9. 验收设计：不是只做一个任务

同一入口覆盖六类验收：信息抽取、计算、条款核验、跨文档比较、报告、多轮追问。再覆盖恶意文档指令、文档越权、Token 缺失、重启恢复、预算耗尽和口径冲突。

先使用合成材料检查可确定的逻辑和边界，再使用允许使用的真实新材料测质量。合成测试通过不等于真实问答正确率。真实评测按公司/文档版本划分开发与测试，避免复用已针对性优化的比赛题作为唯一泛化证据。

必须分别记录：答案正确、计算正确、引用可回取、引用支持结论、拒答适当性、总用量、时延、补查次数和恢复情况。状态机测试检查必要条件，不要求所有模型严格走同一串工具调用。

## 10. 交付与下一次执行起点

本轮交付的是可实施设计、依赖清单、验收清单和已运行的底层测试证据；没有声称已完成 Agent 升级。

下一次编码从批次 A 开始。在发布仓库的开发副本/分支中恢复两项依赖，导入纯函数回归测试，然后建立新 contracts 与检索适配器。保留现有最高分文件不变，不整目录复制历史工作区，不读取或复制私有配置，不直接运行会覆盖旧答案的历史批处理脚本。
