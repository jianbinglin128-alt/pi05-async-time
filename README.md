# pi05-async-time

ActSync Future 动作上下文模型的轻量自适应推理触发调度器。固定预测长度与执行长度，通过近期请求耗时和动作执行耗时自动选择 k，替代手动切换触发位置。

本仓库包含代码、配置、测试及说明。模型权重、虚拟环境、缓存、数据集、实验视频和逐回合日志均不包含在仓库中。上游 OpenPI 说明保存在 [README.upstream.md](README.upstream.md)。

**版本修复提示：初始提交 `8c92507` 的 JAX 模型缺少动作上下文投影与注意力输入。本版恢复训练时的上下文计算路径，并强制完整参数匹配。旧版 LIBERO 结果不能用于评价修复版 Future 调度效果。更新后必须重启服务端和客户端；新客户端会拒绝旧服务端。** 详见 [修复说明](docs/context_fix.md)。

## 调度规则

- 默认 H=10、E=10，候选 k=2..9，初始 k=4。
- 请求时间估计为近期 P95，动作执行时间估计为近期 P20。
- 在候选范围内选择满足 `k * action_time >= request_time + max(2ms, 5% * request_time)` 的最小 k。
- 每个 chunk 开始时冻结决策；需要时立即增大 k，连续三块允许更小 k 后再降低一档。
- 只有一个在途请求。提交给模型的上下文必须全部执行完，结果早到时缓冲，晚到时等待。
- 预测从上下文之后开始，新输出保留前 E 个动作，不跳过前 k 个动作。

若没有候选值满足预算，使用最大 k 并记录不可行。经验分位数规则不构成实时保证或任务质量保证。该机制参考 [ActECC IV-B](https://arxiv.org/html/2603.07949v3#S4.SS2) 的最晚可行请求原则，适配固定动作上下文交接。

可通过 `--stage-aware` 启用来自 `openpi_async_stage` 的 coarse/fine 状态上限：精细阶段请求上限 `floor(E/2)`，粗操作请求上限 `2E`，实际均受 `min(E-1, 9)` 的执行与权重限制。默认 E=10 时实际上限为 5/9。状态由已执行的末端运动历史和本块预测夹爪事件判断，计时调度器在上限内选 k；上限内预算不足时接受等待。默认仍使用原计时模式。配置来源、迁移差异与命令见 [阶段上限说明](docs/stage_k_caps.md)。

## 环境

Linux、NVIDIA GPU、可用的 CUDA 驱动和 Python 3.11。

```bash
git clone --recurse-submodules https://github.com/jianbinglin128-alt/pi05-async-time.git
cd pi05-async-time
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
uv pip install --python .venv/bin/python -r requirements-time-eval.txt
```

按 [LIBERO 官方说明](https://github.com/Lifelong-Robot-Learning/LIBERO) 准备资产、任务定义和初始状态，并配置 `LIBERO_CONFIG_PATH`。调度器默认从 `third_party/libero` 导入 LIBERO；也可通过 `LIBERO_SOURCE=/path/to/LIBERO` 选择已有安装。

`requirements-time-eval.txt` 记录本次验证使用的仿真依赖版本；不要使用旧的 `examples/libero/requirements.txt` 覆盖 Python 3.11 环境，该文件保留的是上游旧环境配置。已验证的主要版本为 JAX 0.5.3、Flax 0.10.2、PyTorch 2.7.1、NumPy 1.26.4、MuJoCo 2.3.7、robosuite 1.4.1。

启动脚本优先使用 `TIME_PYTHON`，其次使用本仓库 `.venv/bin/python`，最后使用 `python3`。已有兼容环境可直接设置 `TIME_PYTHON=/path/to/python`，无需重新安装。

## 模型权重

使用 [Dream-LIN/pi05_libero_async_future](https://huggingface.co/Dream-LIN/pi05_libero_async_future)，Orbax/JAX checkpoint step 29999，对应 30,000 次优化更新。服务端校验权重 manifest 的 SHA256。这里提供的是推理运行时，不包含 Future 模型训练复现配置。

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Dream-LIN/pi05_libero_async_future",
    local_dir="checkpoints/pi05_libero_async_future",
)
PY
```

首次运行可能需要下载上游 tokenizer，`OPENPI_DATA_HOME` 可指定共享缓存。预热期间 JAX 编译可能耗时较长，客户端已关闭 WebSocket 心跳超时。

## 运行

```bash
# CPU 测试，不需要模型权重或 LIBERO 数据。
bash scripts/run_time.sh test 0

# 完整权重上下文检查，需要 checkpoint；不会运行 LIBERO。
bash scripts/run_time.sh check-context 0 \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future" \
  --output "$PWD/results/context_check.json"

# 真实权重上下文检查，需要 GPU；使用固定合成观测，不评估任务成功率。
bash scripts/run_time.sh check-context 0 \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future" \
  --output "$PWD/results/context_check.json"

# 终端 1：服务端，10 次 flow 步，完整未剪枝权重。
bash scripts/run_time.sh server 0 --port 19930 \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future"

# 终端 2：四套件各 1 个任务、每任务 3 个初态，自动 k。
bash scripts/run_time.sh eval 0 --port 19930 \
  --tasks 1 --trials 3 --output "$PWD/results/adaptive_smoke"

# 相同条件的固定 k 对照；输出目录必须不同。
bash scripts/run_time.sh eval 0 --port 19930 \
  --tasks 1 --trials 3 --fixed-k 4 --output "$PWD/results/fixed4_smoke"

# 根据 coarse/fine 阶段限制 k；E 仍为 10。
bash scripts/run_time.sh eval 0 --port 19930 \
  --suites libero_spatial --tasks 10 --trials 3 --stage-aware \
  --output "$PWD/results/spatial_stage_cap59"
```

使用 `--suites libero_spatial` 选择套件，`--candidates 2 3 4` 限制候选范围。`--tasks 10 --trials 50` 对每个选定套件评估 500 回合。输出目录必须是新目录，GPU 与端口需避开其他作业。服务端默认只监听本机。

日志包括运行配置、逐请求耗时、逐动作耗时、每块决策和实际交接等待。覆盖由实际动作就绪时间和上下文执行结束时间计算，不由预算预测标志代替。动作计时排除推理等待；任务计时包含 bootstrap，排除预热、reset、稳定动作、终止后排空及磁盘写入。

## 验证范围

35 项 CPU 测试覆盖时间预算、滞回、不可行处理、真实后台线程、动作顺序、阶段上限和零上下文交接，以及实际 JAX 动作专家的上下文依赖、mask、注意力方向和严格权重加载。CPU 网络测试缩小视觉编码器，使用真实的小型 Transformer；完整已训练权重另外通过 `check-context` 检查。阶段功能尚未完成新的 DGX 闭环成功率评测。

初始版的 28 回合检查以及 DGX Spatial 对照都发生在上下文链路修复前，不能作为本版 Future 模型成功率证据。修复版验证范围见 [修复说明](docs/context_fix.md)。尚未在全新机器上重建整个环境；小规模检查不能作为完整 LIBERO 成功率或独占设备提速结论。

## 关键代码

- [adaptive_k_scheduler.py](packages/openpi-client/src/openpi_client/adaptive_k_scheduler.py)：计时估计和 k 决策。
- [adaptive_future_broker.py](packages/openpi-client/src/openpi_client/adaptive_future_broker.py)：异步请求、上下文承诺与交接。
- [serve_time.py](scripts/serve_time.py)：Future 权重加载和服务端。
- [eval_time.py](examples/libero/eval_time.py)：自适应与固定 k 的 LIBERO 评估。

源码基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。保留原有 [Apache 2.0](LICENSE) 和 [Gemma](LICENSE_GEMMA.txt) 许可；模型、数据及第三方依赖按其各自条款使用。
