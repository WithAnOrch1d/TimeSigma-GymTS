# TimeSigma-GymTS

TimeSigma 是基于 Pi 和 EasyTSAD 的时序异常检测 Research Agent。本仓库供 GymTS 接入：从读取历史经验、规划 Pipeline、执行实验，到保存 ResearchStep 和生成三 Schema 排行榜。

建议先跑不收费的 Memory 示例，再运行单任务 Agent；完整实验使用独立发布的历史资产。所有操作均从仓库根目录执行。

## 先理解架构，再运行当前版本

- **理解架构与研究流程**：阅读 [V1 历史架构与实验报告](docs/demo/TimeSigma_V1_Research_Dossier.html)，了解 Pi、TimeSigma、EasyTSAD 的分工、Evidence、工具调用和研究树。报告对应六方法的 V1，算法数量、实验成绩与未来计划保留当时状态；其中排行榜链接指向配套的 V1 历史结果。
- **查看当前交付版本的实验结果**：阅读 [Core12 三 Schema 历史排行榜](docs/demo/historical-Core12-Leaderboard-3Schema.html)。本仓库交付 Core12；运行方式、支持范围与验收边界以本 README 和当前 Catalog 为准。

HTML 在 GitHub 文件页可能显示源码；克隆仓库后用浏览器打开对应文件即可，页面可离线阅读。V1 与 Core12 的结果分别保留，历史成绩不保证在新硬件或新模型会话中完全复现。

## 安装环境

建议 Linux/WSL、Python 3.12.3、Node.js ≥22.19.0、Git 和 uv。环境由使用者安装，不随仓库分发。

```bash
uv sync --frozen --extra torch
cd pi-host
npm ci
npm run build
cd ..
```

Python 依赖由 `uv.lock` 锁定，EasyTSAD 使用固定 Git revision；Pi 依赖由 `package-lock.json` 锁定。PyTorch 使用2.5.1 CUDA12.1 wheel，可在CPU上加载小型样例；神经方法可能需要GPU及相应NVIDIA驱动。

MatrixProfile 需要独立Python3.9环境：

```bash
uv venv --python 3.9 .runtimes/easytsad-python39
uv pip install --python .runtimes/easytsad-python39/bin/python -e '.[matrix-profile]'
```

该独立环境按声明范围解析依赖，尚无单独的完整依赖锁；交付验收未重新安装或执行MatrixProfile，需要使用者验证后用于实际研究。

## 先跑 Memory 示例：不收费、不训练

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python -B examples/handoff/inspect_memory.py
```

预期输出 `status: PASS`，包括模型加载、Evidence血缘、Memory搜索、比较和两条ResearchStep的读取。无需原始数据、GPU或API key，不写Memory。

`examples/handoff/sample/`附带约212KiB真实数据：AR/OCSVM在TODS/0、234的4个Component，AR/234的2个ResearchStep，相关模型和15条TODS Profile。比较表中未附带的LOF记录保持`missing`，不视为零分。样例的manifest保留原六数据集声明，但只有这6条Evidence；它不是完整Memory，也不应回写原始科学资产。来源和文件校验见`examples/handoff/sample-source.json`。Artifact加载检查不等于数值replay，指标和解释均为原历史记录。

## 跑一次真实 Agent 实验（通用 Core12 入口）

将 `pi-host/.env.example` 复制为 `pi-host/.env`，填写自己的Provider、模型ID和API key。支持现有OpenAI/DeepSeek接口，模型ID须使用账户当前可用的值；不要提交真实`.env`。

```bash
.venv/bin/python scripts/fetch_example_data.py
PYTHONPATH=. .venv/bin/python examples/prepare_research.py \
  --output runs/demo-001 --execute
```

数据脚本从固定上游revision下载TODS，实验执行时不临时获取数据。示例为 `TODS/234/NAIVE`：最多1次实验、1200秒、24次Provider请求、200,000 tokens。`--execute`会调用付费模型并可能训练；省略它只准备scope、上下文和启动命令。若已准备目录，执行打印的启动命令，不要再次prepare到同一路径。

这个通用入口没有CPU算法限制，Agent可选择Core12中符合当前Schema的方法。需要备好所选方法的GPU/MatrixProfile运行环境；小型Memory不会将Agent限制为CPU算法。Agent也可能不提出新实验，或实验没有提升，检查实际记录而非只看进程退出码。

### CPU / GPU 如何选择

| 场景 | 使用方式 |
|---|---|
| 正常 Core12 Agent 实验 | 使用上述通用入口，不加 `--cpu-only`；检测器沿用各自原生设备行为与Runtime要求。 |
| 使用指定 GPU | 配好NVIDIA驱动与CUDA兼容PyTorch，可在同一命令前设置 `CUDA_VISIBLE_DEVICES=0`；这只指定可见设备，不强制CPU方法转为GPU方法。 |
| 无GPU，先验证真实Agent链路 | 使用下方 `--cpu-only` 示例，仅允许AR受控实验。 |
| 生成单任务 Component | 使用后文 `scripts/build_component.py`；默认CPU，GPU执行显式传 `--device cuda`。 |

例如在有GPU的机器上运行同一个Agent入口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python examples/prepare_research.py \
  --output runs/gpu-agent-001 --execute
```

该命令仍由Agent规划方法，不保证一定选择GPU检测器。完整Research和三Schema AIOPS也使用正常Core12执行路径，见后文；本次交付重新验收的是CPU链路，不代表全部GPU方法已在新环境实测。

```text
runs/demo-001/
├── scope.json / context.json / launch.json
├── memory/agent-memory/      # 本次独立可写副本
├── memory/profiles/
└── run/
    ├── research-run.json    # 状态、预算与用量
    ├── provider.jsonl       # Provider事件
    ├── sessions/            # Pi对话与工具事件
    └── experiments/         # 配置、反馈、Artifact与官方评价
```

原样例不变。保留失败目录，不通过删除记录重置预算。ResearchStep是科学转移记录，完整对话还需查看`sessions`和Provider事件。

### 无GPU时：可选的CPU验收示例

```bash
PYTHONPATH=. .venv/bin/python examples/prepare_research.py \
  --output runs/cpu-agent-001 --cpu-only --execute
```

仍使用真实Pi/Provider、Memory、preview、训练、评价、ResearchStep提交。这个**验收包装器**只允许经过preview的AR、无ensemble/额外operators的Pipeline进入执行，不改生产Core12 Catalog；LLM自行提出一个受控AR参数/context-length假设。最多1次实验，预算同普通示例。它会调用付费模型。Agent若主动结束或仅复用旧Evidence，不能声称完成了新训练全链路；检查实际`research-run.json`与新ResearchStep。该验收不代表全部Core12或GPU路径已验证。

`--cpu-only`只定义在`examples/prepare_research.py`，通过`examples/cpu_research.mjs`包装现有Agent；它不是生产Research CLI的通用设备开关，也不会修改Core12 Catalog。

## 接入算法与生成单任务 Component

接入新方法需要开发适配，不能只往Catalog里加名称。可参考仓库已有的`timesigma/algorithms/isolation_forest/`：

1. 固定上游源码/许可和依赖；在`pyproject.toml`声明依赖并同步`uv.lock`，需要特殊Python环境时声明独立runtime。不要提交本机环境。
2. 实现或复用EasyTSAD `BaseMethod`接口及原生`config.toml`，保留训练、验证、停止和评分语义。禁止把不同curve拼起来伪造共享Schema。
3. 在`timesigma/algorithms/catalog.py`声明`AlgorithmSpec`：模块、参数约束、Schema资格、原生selection、Artifact类型和score support。资格未完成时保持`agent_available=False`；这不阻止本入口生成审阅候选，但禁止发布。
4. 按生命周期补齐`algorithms/checkpoint.py`或`artifacts.py`、`benchmark/artifacts.py`、`benchmark/input_contract.py`中实际需要的绑定。现有Artifact类型不足时先解决合同，不能伪装成其他类型。Agent参数空间还涉及`research/pipeline_space.py`、`pipeline.py`和`semantics.py`，须检查方法参数及context-length映射。
5. 分别验证真实支持的Schema、原生选模、独立进程加载/replay和官方八指标，再审阅开放Agent资格。一个任务通过不代表全部Schema和数据集已完成qualification。

`scripts/build_component.py`提供**一个canonical任务**的顺序执行入口。NAIVE是一条curve；ALL_IN_ONE/ZERO_SHOT是一整个原有dataset planning unit。它从指定的已有Component继承任务，使用新方法的完整官方默认配置及原Memory的seed/revision/split；不自动改默认参数或生成新split。共享Schema必须传入该Schema的真实参考Evidence，样例只有NAIVE。

先创建独立工作Memory并下载TODS；下面的IsolationForest例子生成样例中缺失的Component，不修改附带样例或历史归档：

```bash
.venv/bin/python scripts/fetch_example_data.py
mkdir -p runs/component-demo
cp -a examples/handoff/sample/. runs/component-demo/
.venv/bin/python scripts/build_component.py \
  --memory runs/component-demo/agent-memory --data datasets/easytsad \
  --reference component/AR/naive/TODS/0 --method IsolationForest \
  --output runs/component-demo/attempt-001
# 核对dry-run后，用相同参数加 --execute 开始一次native训练。
.venv/bin/python scripts/build_component.py \
  --memory runs/component-demo/agent-memory --data datasets/easytsad \
  --reference component/AR/naive/TODS/0 --method IsolationForest \
  --output runs/component-demo/attempt-001 --execute
# ADMISSION_READY后审阅candidate，再显式发布到上述工作Memory。
.venv/bin/python scripts/build_component.py \
  --output runs/component-demo/attempt-001 --publish
```

默认每阶段600秒、4GiB RAM、1 CPU、CPU设备；可在运行前用`--seconds-per-stage`、`--memory-mb`、`--cpu-cores`、`--device cuda`明确设置。保留10GiB磁盘余量。默认不训练；`--execute`执行train→fresh-process replay→官方评价→parser/Artifact校验，生成`candidate/`。`--publish`检查候选完整性后原子发布，沿用原Evidence格式。已有同任务同默认配置且Artifact有效时返回`SKIP_EXISTING`，不重训/覆盖。不同配置占用同一ID时拒绝覆盖。

`plan.json`、阶段日志/资源结果、`replay.json`和`status.json`留在attempt目录，科学分数/评价和必要血缘随candidate保存。失败保留原attempt；修复工程原因后使用新目录，参数和split不应因低分改变。本入口没有批量队列或自动重试，也不代替新算法qualification审阅。不要与其他Memory写入器同时发布。

## 完整实验资产与数据

完整科学资产通过本仓库的 [Releases 页面](https://github.com/WithAnOrch1d/TimeSigma-GymTS/releases)分包分发，不放入Git历史。选择 `core12-assets-20261013` 版本，在 **Assets** 中下载全部分片、`asset-manifest.json` 和 `SHA256SUMS`。代码使用本仓库 `main` 分支；Release自动附带的 `Source code (zip/tar.gz)` 是标签时的源码，不是历史科学资产包。该版本仅在全部上传并校验通过后公开；若页面尚未显示该版本，表示资产仍在准备中，可先运行上面的 Memory 示例。

资产清单摘要见`experiments/asset-release.json`：约62.8GiB，包含9,561 Component、407 ResearchStep、Profiles、关联模型，以及原Research会话和AIOPS计划、分数、评价、冻结源码。原始数据集另外从固定版本获取，须遵循上游使用条款。

下载全部分片、`SHA256SUMS`和清单后，在空间充足的实验机器上执行：

```bash
sha256sum -c SHA256SUMS
cat timesigma-assets.tar.gz.part-* > timesigma-assets.tar.gz
.venv/bin/python scripts/asset_bundle.py unpack \
  --archive timesigma-assets.tar.gz --output /mnt/experiment-assets
.venv/bin/python scripts/fetch_example_data.py --full
```

`/mnt/experiment-assets`是示例路径，可替换为自己的磁盘目录。解压目标必须不存在，程序拒绝覆盖并逐文件验证哈希。需预留压缩包、解压资产和后续实验空间；不改写原Evidence、checkpoint或历史回执。

### 查看或重新生成原排行榜

不下载完整资产也可先用浏览器打开附带的 [Core12历史排行榜](docs/demo/historical-Core12-Leaderboard-3Schema.html)。下载资产后可以重新核对原官方结果并生成HTML，无需LLM或训练：

```bash
.venv/bin/python scripts/render_history.py \
  --assets /mnt/experiment-assets \
  --output runs/history-view/TimeSigma_AIOPS_Leaderboard_3Schema.html
```

程序核对本地八指标、公共快照和排名，沿用保存的规划统计，不访问旧服务器路径。重新生成的HTML不要求与原HTML逐字节一致。

### 继续非 AIOPS Research（可选）

可以直接使用已有Component，不必重训9,561项。继续研究应保留只读归档，另建工作Memory；支持reflink的文件系统可减少起始复制空间，否则需要额外磁盘空间。

```bash
mkdir -p /mnt/research-work
cp -a --reflink=auto /mnt/experiment-assets/runs/agent-memory /mnt/research-work/
cp -a --reflink=auto /mnt/experiment-assets/runs/profiles /mnt/research-work/
PYTHONPATH=. .venv/bin/python scripts/run_research.py prepare \
  --data datasets/easytsad --memory /mnt/research-work/agent-memory \
  --output runs/research-001 --tasks experiments/research-tasks.json
PYTHONPATH=. .venv/bin/python scripts/run_research.py run \
  --output runs/research-001 --execute
```

`research-tasks.json`保留原36个非AIOPS任务及成员顺序，每项最多12次实验。48次Provider请求/480,000 tokens等是原初始预算模板，不是历史恢复后的累计预算，也不保证足够执行12次实验。使用者在prepare前确认预算，或另存缩小的任务清单。新研究从407条最终Memory继续，不是原199条起点的精确复跑。

任务顺序执行，成功项跳过；失败/中断保留状态和用量，按现有CLI合同进行预算保持的恢复，不自动无限重试。不会为凑数量生成Evidence，也不会自动启动AIOPS。

### 新 AIOPS 三 Schema → 新排行榜

使用原最终Memory，或新Research完成后的工作Memory；运行期间不要同时修改它。`experiments/aiops-*.json`保留原成员、seed、预算、补充算法和公共快照。确认本机`.env`与配置的Provider字段一致：其中模型是历史值，更换模型或预算应在prepare前明确设置，并视为新的实验条件。

```bash
PYTHONPATH=. .venv/bin/python scripts/run_delivery.py prepare \
  --data datasets/easytsad \
  --memory /mnt/experiment-assets/runs/agent-memory \
  --output runs/aiops-001
PYTHONPATH=. .venv/bin/python scripts/run_delivery.py run \
  --output runs/aiops-001 --execute
```

一次启动按顺序完成：三Schema全部规划并封闭 → TimeSigma评分 → 冻结的补充算法评分 → 全部分数冻结 → 官方八指标评价 → HTML。NAIVE为29曲线，ALL_IN_ONE为完整29曲线，ZERO_SHOT保留原15训练/14 held-out成员。标签只供官方评价，AIOPS结果不写入研究Memory，不因结果不好重新规划。

补充默认算法为NAIVE的IsolationForest/DCdetector/OCSVM/LOF，以及共享Schema的IsolationForest/DCdetector；LOF/OCSVM共享保持N/A。公共快照已有算法的成绩保留原值，不是所有公共算法都在本机重训。

输出在 `runs/aiops-001/TimeSigma_AIOPS_Leaderboard_3Schema.html` 和 `summary/three-schema-summary.json`。重复启动保留已完成阶段。修复工程故障后可显式加`--retry-failed`，仍受冻结的执行/评价次数限制；规划失败不自动获得新Provider预算。

## 代码结构与 GymTS 接入

Pi负责模型交互、Agent循环、会话、工具调用和上下文；TimeSigma负责研究语义与Memory；EasyTSAD负责原生训练和评价。

| 入口 | 作用 |
|---|---|
| `pi-host/src/research-cli.ts` | 单次Research会话、预算和运行状态 |
| `timesigma/bridge/` | Pi与Python工具之间的JSONL协议 |
| `timesigma/profiles/` | train段Profile，共享Schema仅聚合训练成员 |
| `timesigma/research/` | Memory检索/比较、Pipeline预览、实验与ResearchStep提交 |
| `timesigma/algorithms/`、`benchmark/` | Catalog、原生检测器生命周期、Artifact及官方八指标 |
| `timesigma/delivery/`、Pi `delivery-cli.ts` | 一次性规划、执行、分数冻结和评价 |
| `timesigma/method.py` | EasyTSAD的TimeSigma执行适配 |
| `examples/`、`experiments/`、`scripts/` | 可运行样例、完整实验配置和启动工具 |

Core12：AR、AE、AnomalyTransformer、MatrixProfile、LOF、OCSVM、LSTMADalpha、LSTMADbeta、FITS、FCVAE、IsolationForest、DCdetector。NAIVE支持12个，共享Schema支持10个；Ensemble限于已有白名单。

GymTS可将`Describe`映射到Catalog、Schema和工具协议，将`Start`封装为一次`research-cli`调用。记录层读取会话、工具事件、官方评价和Evidence血缘；Python SDK、Gymnasium环境、Simulator、SFT/RL转换和reward尚未实现。

未包含历史批量Component调度器和优化诊断；完整实验可导入现有Component资产，也可用上述单任务入口增量补齐。`third_party`仅保留条件导入所需的小型adapter/runtime定义，不含外部环境，也不启用OmniAnomaly。

## 路径、结果与验证范围

- 配置使用相对资源路径；prepare根据`--data`、`--memory`、`--output`生成本机路径和绑定。冻结后不移动运行目录或替换源码/输入；换机器应从输入资产创建新run，不能批量替换历史回执中的路径。
- 历史绝对路径仅供溯源。Evidence、配置、指标和历史解释保持原样；Profile相似不是算法收益可迁移的证明，历史解释也不等于因果结论。
- 当前源码包含原实验之后的事实处理修复。LLM、硬件和随机性会影响重跑结果；复现流程不承诺相同指标。历史冻结源码保存在资产包的原执行根中。
- AIOPS已被查看，后续开发不能声称项目级严格未见测试。HTML是本地官方评价与冻结公共快照的比较，不是官方榜单提交。

测试命令：

```bash
.venv/bin/python -m pytest tests -q
cd pi-host
npm run typecheck
npm test -- --no-file-parallelism
npm run build
cd ..
# 可选：生成上传清单并在新目录中验证交付文件。
.venv/bin/python scripts/check_release.py --output runs/release-check --test
```

已验证Python91项、Pi65项、typecheck/build、Memory样例和真实模型资产迁移。服务器从GitHub独立克隆、全新Python/Node依赖安装和固定数据下载通过；IsolationForest/TODS/NAIVE/0已完成真实训练、fresh-process replay、官方八指标、候选校验、隔离Memory发布和existing skip。共享Component的成员继承已做边界测试，本次没有重新训练共享或GPU任务。服务器真实三Schema及Research上下文prepare、历史HTML重建亦已通过。CPU Agent真实验收完成：TODS/234上由LLM提出AR p=16→4，完成preview、训练、官方八指标、诊断和1条新ResearchStep提交；7次Provider请求、103,577 tokens，达到预定1次实验后正常结束，无未完成项。结果只写隔离验收Memory。这是可运行性检查，不是泛化性能证明。MatrixProfile独立环境未重新验证。

环境、原始数据、大型科学资产和密钥不进入Git。使用者安装环境、提供自己的模型账户；上游代码、数据和字体保留各自来源及许可，本仓库尚未另行指定项目开源许可证。
