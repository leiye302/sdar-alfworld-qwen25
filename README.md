# SDAR on ALFWorld: Qwen2.5-1.5B / 3B, 8×A100 80GB

独立运行 [SDAR: Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) 的 ALFWorld 对比实验。直接复用 [官方代码](https://github.com/ZJU-REAL/SDAR)，锁定 `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`。**两个尺寸分别从原始 Instruct 基模开始，各训练150 updates；每个实验使用完整八卡，默认先1.5B、再3B顺序运行。**

本版本面向 **8×A100 80GB**，恢复官方八卡、rollout TP=2、vLLM显存比例0.6、CUDA graph和原生非fused forward。不是把四卡的全局batch翻倍，也不是两个尺寸各四卡同时运行。较早的四卡版本不能直接用八卡配置续训；见第6节。

| 实验 | 初始化 | 训练 |
| --- | --- | --- |
| SDAR 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 官方 GRPO＋sigmoid-gated SDAR，150 updates |
| SDAR 3B | `Qwen/Qwen2.5-3B-Instruct` | 完全相同的配方，150 updates |

这是 SDAR 论文主方法的入口 `examples/sdar_trainer/run_alfworld_3b.sh`，**不是** `grpo_opsd_trainer` 的 beta=0 基线，也不是后续的 RetireOPD。1.5B 不需要改模型架构或算法，只换官方模型 ID 和固定 revision。没有 Route-SFT、RETHINK 注册、Analyzer、局部 hint 或付费 API。

## 1. 方法与固定配方

Student 正常交互生成轨迹，GRPO 使用整条轨迹的环境奖励。同一 Actor 权重的无梯度 Teacher，在原 Student 前缀上额外获得官方 ALFWorld 静态 skill，对 Student 已生成的 token 重新评分。Teacher 随 Actor 更新；reference KL 的参考模型保持固定。

$$
\Delta_t=\log\pi_\theta(y_t\mid H_t,\mathrm{skill})-\operatorname{sg}[\log\pi_\theta(y_t\mid H_t)],\qquad
g_t=\operatorname{sg}[\sigma(5\Delta_t)].
$$

$$
\mathcal L=\mathcal L_{\mathrm{GRPO}}+0.01\mathcal L_{\mathrm{SDAR}}+0.01\mathcal L_{\mathrm{reference\ KL}}-0.001\mathcal H.
$$

式中 SDAR 仍是官方 sampled-token、非负 sigmoid gate 的辅助目标；不是全词表 KL 或独立固定 Teacher 的 OPD。实际实现以锁定源码为准。非法动作惩罚仍按原实现施加，系数0.1。

| 项目 | 固定值 |
| --- | --- |
| 训练池 | 原生 ALFWorld `train` 的3553个可用任务，由官方环境采样 |
| 每 update | 16个任务 × 8条轨迹 = 128条 episode；决策行数随轨迹长度变化 |
| GRPO / SDAR | `adv_estimator=grpo`；SDAR coefficient=0.01；gate beta=5；`skill_all=false` |
| 学习率 / PPO | 1e-6；mini-batch256；每卡 micro32；PPO epoch1；token-mean |
| Reference KL / entropy | 0.01 low_var_kl；entropy coefficient0.001 |
| 环境 / 生成 | max steps50；history2；prompt2048；response512；训练temperature1 |
| 验证 | 初始及每10 updates；固定seen140，一任务一次rollout，temperature0.4 |
| 保存 | 每20 updates及最终150；保留最近两份完整断点，最终150包含在其中 |

两种尺寸共用 [experiment.json](configs/experiment.json)，除模型 ID / revision、实验名称和输出路径外不改变配方。模型版本记录在 [models.json](configs/models.json)。

## 2. 八卡服务器准备

要求 Linux x86-64、Python3.12、8×A100 **80GB**、兼容CUDA12.8的驱动。建议64个可用逻辑CPU、256–512GiB主机内存、至少300GiB空闲磁盘，400GiB更稳。磁盘包含两个尺寸各最近两份可续训断点、公开基模、环境、缓存及最终导出和第三份存盘峰值；这是部署预算，不是实测最小值。A100 40GB不属于本配方已配置的硬件；本仓库不远程自动占卡。

```bash
git clone https://github.com/leiye302/sdar-alfworld-qwen25.git
cd sdar-alfworld-qwen25

# 也可使用已准备好的 Python3.12 conda 环境。
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

# 仅需 Git 与标准库即可先准备官方源码，不下载权重。
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 编辑获配的八个 GPU 编号、CPU 数量和 ALFWORLD_DATA。
set -a
source .env
set +a

# 阅读两种基模的 LICENSE 后下载固定版本；不需要作者的文件。
python src/setup.py --download-model both
# 官方下载器读取 ALFWORLD_DATA。准备 json_2.1.1 和 logic；无需重新采集轨迹。
alfworld-download
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
```

`flash-attn` 使用官方安装方式，预编译 wheel 不匹配时需要 CUDA Toolkit / nvcc 现场编译。依赖约束防止上游安装过程升级 torch / vLLM。需要时可设置自己的 `HF_ENDPOINT` 或 Hugging Face 凭据；模型 ID 和 revision 仍必须与锁文件一致，不能换成不同模型。

本仓库不捆绑数GB模型或ALFWorld资源；所有资源都有公开获取命令，**不依赖作者电脑、私有服务器、外接硬盘或私有数据集**。静态 skills 随锁定的官方源码下载。

## 3. 开跑

先查看两种尺寸的完整配置，不占GPU、不读大权重、不启动训练：

```bash
bash scripts/train.sh --size 1.5b --dry-run
bash scripts/train.sh --size 3b --dry-run
```

刷新 GPU 实时占用，确认八卡归自己使用。在 `tmux` / 作业调度器中执行：

```bash
# 1.5B150 → 导出；随后3B150 → 导出。每个实验都使用SDAR_GPUS指定的八张卡。
bash scripts/run_both.sh
```

也可独立运行某个尺寸：

```bash
bash scripts/train.sh --size 1.5b
bash scripts/train.sh --size 3b
```

**不要在同一组八张卡上同时执行上面两条命令。** 没有外层 `torchrun`：由SDAR/Ray启动八个FSDP rank，rollout为四个TP=2副本。八卡共同消费每update的128条episode，不是每卡128条。每个实验的Ray使用独立短路径临时目录和 `address=local`；不连接其他实验的集群，也不要执行整机 `ray stop`。每个运行目录有进程锁，防止重复提交同一输出。

第一次真实运行应检查：八个rank、TP=2的rollout、环境交互、非空奖励、Teacher gap和gate、有限grad norm、Actor backward/update及一次完整存盘。仓库中的CPU测试不能替代这一检查。不要因初期success为0或gate变化就改方法；若遇到OOM，先保留日志并核对实际模型、GPU显存和resolved_config，任何内存配置变更均须记录，不能悄悄改变global batch、group size、loss或技能来源。

## 4. 评测协议与主实验比较

固定任务、游戏文件SHA、顺序和每任务环境seed均在 [seen140_tasks.json](configs/seen140_tasks.json)，与 [RETHINK 1.5B 主实验仓库](https://github.com/leiye302/rethink-alfworld-qwen25-1.5b) 的 seen140 文件相同。

- split=`valid_seen`，140个任务，seed=`2026 + task_index`。
- 每任务1条rollout；最多50步；history2；2048-token prompt、512-token response。
- temperature0.4、top_p1、top_k=-1；采样开启。
- 最终比较update150，不从验证集挑选最佳checkpoint充当最终模型。
- Student 评测没有 Teacher skill、Analyzer 或 hint；完整保留生成文本和token IDs用于reasoning统计。

**这是配合主实验的固定seen140比较协议，不宣称与论文原始128条验证设置相同。** 官方动作解析器保留原样；`strict_protocol_rate` 只是额外观测指标，不会改变执行、奖励或样本过滤。若另一实验改用了更严格的执行解析器，必须在比较报告中注明，并在需要严格同协议结果时对两边使用同一评测执行器复测。

训练本身已在update0/10/.../150运行真实环境验证。每次保存：

```text
runs/sdar_1.5b/                         # 3B 对应 sdar_3b
  binding.json                         # 模型、源码、配方和任务集绑定
  resolved_config.yaml
  metrics.jsonl                        # 原生全部训练/验证指标
  training_pool.json                   # 训练任务身份，与seen140互斥
  evaluation/update_000/summary.json    # 成功数/140、reasoning与格式统计
  evaluation/update_000/decisions.json  # 逐步文本/token IDs/原生动作有效标志
  evaluation/update_150/summary.json
  checkpoints/global_step_150/          # 全rank模型、optimizer、scheduler/RNG、dataloader
  hf150/                               # 运行 run_both.sh 后的最终推理导出
```

单独导出和重新评测：

```bash
python src/run.py export --size 1.5b
bash scripts/eval.sh --size 1.5b --model "$PWD/runs/sdar_1.5b/hf150" \
  --run-dir "$PWD/runs/sdar_1.5b_final_eval"
# 3B替换size和模型/输出目录；同样使用八张已空闲的GPU。
python src/report.py
```

`report.py` 读取实际结果生成 `results/seen140.csv`，有matplotlib时另生成PNG；没有预填成绩。只需要评测时，先完成环境/资源准备即可，不需要重训。

## 5. 中断与断点

确认旧进程退出且最新断点完整后：

```bash
bash scripts/train.sh --size 1.5b --resume
# 或 --size 3b --resume
```

使用官方原生续训机制，保存权重、optimizer、scheduler、rank RNG和dataloader；启动前检查全部八rank文件及模型/配方绑定。该机制**不保存每个TextWorld环境的游戏迭代器**，所以不保证中断续跑与不中断训练逐位相同。若首个20步断点尚未完成，不能把残缺目录当作完整断点恢复。默认新训练拒绝覆盖已有输出。

两尺寸顺序运行中断时，先用上面的单尺寸命令恢复尚未完成的训练，再运行该尺寸的export；下一尺寸用独立train命令启动。`run_both.sh`是首次执行队列，不会覆盖已有输出或自动把失败任务从0重跑。

## 6. 与官方8卡脚本的差异

原生GRPO优势、SDAR loss、detach、Teacher构造、skill检索、奖励、prompt、动作解析均保持官方源码；12个核心文件的SHA锁在 [upstream.json](configs/upstream.json)。训练循环直接继承官方 `SkillSDRayTrainer.fit`。

| 项目 | 官方3B脚本 | 本对比入口 |
| --- | --- | --- |
| Backbone | Qwen2.5-3B-Instruct | 同3B，另加官方1.5B |
| GPU / rollout TP | 8卡 / TP2 | 相同；每种尺寸都用八卡 |
| vLLM显存 / cache / graph | 0.6 / 不释放 / graph开启 | 相同 |
| Log-prob计算 | 普通forward | 相同，use_fused_kernels=false |
| 环境进程 | 每环境一个Ray actor | 每16个独立原生环境共用一个actor，环境数/seed/结果顺序不变 |
| 验证 | 128条、每5步 | 固定seen140、每10步，与主实验一致 |
| 存盘 / 日志 | save=-1；console+wandb | 每20步及150、最近两份；console+本地JSON |

`patches/runtime.patch` 只改环境调度/固定验证及trainer导入，另复制一个固定任务选择助手。**不把RETHINK方法的大补丁带进SDAR。** 启动时逐一核验未修改的核心文件和修改过的运行文件，防止误用旧本地实验代码。

独立配置只覆盖模型、资源、固定评测、日志和存盘等字段；单独测试将关键算法/rollout字段逐项对照锁定的官方3B启动脚本。完整适配清单和首轮验收见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。此前四卡版本的recipe/world-size绑定与本版不同，禁止将四卡断点冒充本版八卡续训断点；如需迁移应单独制定并验证重分片流程，本仓库不隐式执行。

## 7. 验证范围与许可

测试覆盖干净源码准备、两种尺寸配置、八卡分配、官方启动脚本逐字段对照、固定任务/版本拒绝、分组环境seed和返回顺序、完整八rank断点检查、以及官方SDAR loss的Teacher依赖与detach梯度。GitHub Actions在全新Linux检出上执行这些检查。**本仓库不附带训后成绩；A100的CUDA/NCCL/vLLM训练尚需运行者进行真实首轮确认。**

包装代码与SDAR源码采用Apache-2.0，保留第三方声明。**Qwen2.5-1.5B-Instruct是Apache-2.0；Qwen2.5-3B-Instruct采用Qwen Research License**，不能把3B权重一概称为Apache许可。ALFWorld资源遵循其来源许可。详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
