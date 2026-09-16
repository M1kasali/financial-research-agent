# 面向金融长文本的证据推理 Agent

面向金融长文档问答，围绕证据检索、上下文压缩与条款推理，支持模型驱动的定向补查与答案来源追溯。

相关团队参赛方案取得 **AFAC 2026 赛题四 B 榜第 35 名**（团队确认）。本仓库整理可复用组件和后续 Agent 扩展；比赛名次不代表扩展能力的评测结果，也不表示本仓库完整复现最高分提交流程。

技术栈：Python 3.12 · Qwen · BM25/BM25F-lite · RAG · pytest

## 核心能力

- **检索与证据覆盖**：保留原方案 Doc-first、字段加权和文档覆盖组件。当前自由问答入口在指定文档范围内检索，不宣称完整接入原全库盲检流程。
- **上下文压缩**：复用问题相关的连续原文摘录与选项证据窗口，按字符预算组织上下文；字符限制与实际 Token 统计分开记录。
- **受限补查闭环**：模型识别证据缺口并提出查询，程序检查范围、重复和预算，追加证据后重新回答与复核；默认最多一次补查，无新增证据时停止。
- **来源核验**：引用绑定文档版本与原文位置。可选 `span_id` 引用模式由程序回填原文，再进行引用检查与单独模型复核，不等同于保证语义正确。
- **财务工具**：对受支持的年度财报核查期间、口径和单位，使用确定性计算返回原值、公式及来源；不支持任意版式或通用投研任务。

参赛阶段的跨文档条款比较与逐选项推理设计见[核心接入说明](docs/v45-core-integration-2026-09-14.md)及[原团队仓库](https://github.com/XxJjTt6/afac2026-financial-longtext-agent-team)。当前工程入口与原提交的关系见[来源说明](NOTICE.md)。

## 快速开始：无需比赛数据或 API 密钥

推荐 WSL/Linux、Python 3.12。以下命令仅安装依赖并运行离线程序，不调用模型。

```bash
git clone https://github.com/M1kasali/financial-research-agent.git
cd financial-research-agent
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
```

仓库当前私有，克隆需要已获授权的 GitHub 账号。`requirements.lock.txt` 是开发环境版本快照；常规安装使用 `pyproject.toml`。

### 1. 检索与压缩真实执行，材料为合成示例

```bash
.venv/bin/financial-agent catalog --chunks data/example/chunks.jsonl

.venv/bin/financial-agent prepare \
  --chunks data/example/chunks.jsonl \
  --query '比较甲乙合同的提前终止通知期限及例外条件' \
  --doc example_contract_a --doc example_contract_b
```

输出为检索结果和压缩证据包，不是模型生成的最终答案。`data/example/` 内文档、公司与条款均为虚构，不来自比赛材料。

### 2. 离线问答协议演示

```bash
.venv/bin/python scripts/run_core_answer_smoke.py
```

脚本使用合成数据和预设模型响应，禁用网络，展示提取、拒答、澄清、计算转交、错误引用拒绝、预算终止和复核不通过七种情况。输出目录在 `experiments/runs/` 下；这是程序集成检查，**不是模型能力或准确率评测**。

### 3. 查看受限补查与引用校验测试

```bash
.venv/bin/python -m pytest -q tests/test_core_research.py tests/test_citation_spans.py
```

以上测试也使用合成输入及模拟模型，不需要密钥。

## 使用自己的文档

将有权使用的材料转换为与 `data/example/chunks.jsonl` 相同的 JSONL 分块结构，放入已忽略的 `data/local/`。先通过 `catalog` 查看文档 ID，再用 `prepare` 检查证据。

当前 `answer` 默认禁用模型。只有显式添加 `--execute` 和一个全新的 `--output-dir` 才读取本项目 `.env` 并发送问题及选中证据至配置的 Qwen 服务；不要将敏感材料放入未获批准的模型调用。

如需在线问答，可手动将 `.env.example` 复制为 `.env`，填写自己的密钥，然后参考命令帮助：

```bash
.venv/bin/financial-agent answer --help
```

当前 CLI 显式限制为 `qwen3.7-flash` 与配置的官方 HTTPS 端点，使用前确认账号可用性。`--citation-mode span_id` 启用片段引用；`--no-follow-up` 关闭补查与财务工具转交，用于单次问答对照。费用限制基于保守估算预留，不是服务端账单硬上限。

## 架构与边界

默认问答入口：`CoreResearchEngine` → 指定范围检索与证据窗口 → `CoreAnswerEngine` 回答/引用检查/复核 → 按需补查或财务工具 → JSON 与 Markdown 报告。

状态、证据和预算由程序维护；模型提出受限动作，不允许任意工具、网络操作或代码执行。旧 `ResearchRuntime` 的任务图与 SQLite 恢复保留在仓库中，但不是当前 `answer` 默认入口，也未完整接入新闭环。

- [计算与补查闭环](docs/core-research-2026-09-14.md)
- [问答与引用核验](docs/core-answer-2026-09-14.md)
- [可见原文片段 ID](docs/citation-spans-2026-09-14.md)
- [财务工具边界](docs/financial-workflow-2026-09-14.md)
- [小规模真实开发对照](docs/core-comparison-2026-09-14.md)
- [引用模式对照及失败记录摘要](docs/live-citation-comparison-2026-09-14.md)
- [首次私有发布检查](docs/publication-checklist.md)

日期文档保留各阶段结果和限制，测试数量按当时版本记录；当前验收以最新测试为准。已知开发案例不等于独立准确率；没有公开标准答案的 B 榜不能由本地案例替代评测。结构化事实记忆、通用规划、生产服务与完整最高分复现均不作为已完成能力。

## 仓库内容与发布范围

```text
src/financial_agent/   当前应用与受限执行流程
vendor/               保留来源记录的团队算法组件
tests/                离线测试与迁入回归
scripts/              合成演示、检查及本地实验工具
data/example/         可随仓库分发的合成示例
configs/              本地数据配置示例与回放目录索引
docs/                 技术说明、来源清单与阶段结果
```

密钥、原始 PDF、比赛题库/语料、索引、运行响应、数据库、人工评测草案和个人求职材料不随首次提交上传。`configs/datasets.example.json` 保留历史数据路径示例，需要本地自行配置，不是下载入口。

`scripts/demo.py` 是本机历史真实运行的哈希校验回放，依赖未上传的 `experiments/runs/`，**克隆仓库后不能直接运行完整回放**；仅 `--list` 可列出案例。它不会在缺记录时自动请求模型。其他依赖比赛语料、人工草案或旧项目路径的实验脚本也不是快速开始入口。

当前为私有工程仓库，未授予开源许可证；公开前须另行确认团队代码及数据授权，详见 [NOTICE.md](NOTICE.md)。
