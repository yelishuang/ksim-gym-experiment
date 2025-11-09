# Repository Guidelines

## 项目结构与模块组织
- `train.py`：主训练脚本，封装任务定义、奖励函数与评估循环；新增算法时可在本文件或独立模块实现并在此导入。
- `convert.py`：把 `humanoid_walking_task/run_*/checkpoints/ckpt.bin` 转 `*.kinfer`，供真实机器人或 `kinfer-sim` 调用；同一目录下可保存奖励曲线以便核对。
- `train.ipynb`：用于交互式实验，保持单元清晰并在提交前清理输出；需要示例时复制脚本中的最小片段。
- `humanoid_walking_task/run_*`：包含 TensorBoard 日志、渲染视频与模型快照；长跑实验建议移动到 `humanoid_walking_task/run_archive/`，仓库中只保留比较用的少量运行。
- `requirements.txt`、`pyproject.toml`、`Makefile` 管理依赖和常用命令；自定义任务宜放在 `benchmark/`（如新建）并在 `tests/benchmark/` 中一一对应。
- 虽暂未创建 `tests/` 目录，但请在新增功能时建立 `tests/<module>/test_<feature>.py`，方便 `pytest` 自动发现并与模块树保持一致，必要时添加 `__init__.py` 以启用命名空间包。

## 构建、测试与开发命令
- `make install`：安装运行时依赖；切换 GPU 驱动后请重新执行以获取正确的 `jax[cuda12]` 轮子。
- `make install-dev`：安装 `ruff` 与 `mypy`，便于本地 lint/类型检查。
- 典型训练命令：
```bash
python -m train max_steps=100 seed=0 wandb_mode=offline
```
- 浏览或调试：
```bash
python -m train run_mode=view load_from_ckpt_path=humanoid_walking_task/run_0/checkpoints/ckpt.bin
tensorboard --logdir humanoid_walking_task
```
- 转换与部署：
```bash
python -m convert humanoid_walking_task/run_0/checkpoints/ckpt.bin assets/model.kinfer
```
- `make notebook` 在 0.0.0.0:8888 启动 Jupyter，首次使用请设密码或通过 SSH 隧道访问。

## 编码风格与命名约定
- 目标 Python 3.11+；模块、函数使用 snake_case，类用 PascalCase，配置常量大写下划线。
- `ruff` 控制格式与 lint，行宽 120、默认双引号；`make format` 依次执行 `ruff format` 与 `ruff check --fix`。
- `pyproject.toml` 启用 `ANN`、`D`、`PL*` 等规则并采用 Google Docstring，请为公共 API 补全文档字符串和示例。
- `mypy` 设置为严格模式：避免 `Any`，所有公共函数声明返回类型；与第三方库交互需在 overrides 中标记 `ignore_missing_imports`。
- 复杂逻辑分成纯计算和副作用函数，便于复用与测试；共享超参集中到单一配置对象，避免魔法数字散落各处。

## 测试指南
- `pytest` 为默认框架，配置 `-rx -rf -x -q --full-trace`，失败即停止以节省 GPU 时间。
- 新功能的测试放在 `tests/<module>/test_<behavior>.py`；慢速用例加 `@pytest.mark.slow`，CI 默认执行 `pytest -m "not slow"`。
- 训练或转换相关测试务必使用极小 `max_steps`、固定随机种子与 `tmp_path` 输出，例如 `tmp_path / "humanoid_smoke"`，避免污染真实 run。
- 提交前运行 `pytest`、`make static-checks` 并在 PR 中贴出关键命令及结论，若跳过测试需说明理由与后续补救计划。

## Commit 与 PR 指南
- 推荐祈使句 + 可选 scope 的提交信息，例如 `trainer: refactor loss logging` 或 `docs: add kinfer checklist`。
- 单次提交聚焦单一主题，包含代码、文档与测试；删除或移动文件需在描述中注明替代方案与迁移步骤。
- PR 描述需覆盖动机、实现摘要、验证步骤、潜在风险，并使用 `Fixes #123` 关联 Issue；跨仓库依赖请列出版本。
- 涉及可视化的改动请上传 `humanoid_walking_task/run_x/video.mp4` 或截图，并说明采集命令；大文件放入 Git LFS。
- 请求评审前确认 `make format`、`make static-checks`、`pytest`、最小训练冒烟（例如 10 步）均通过。

## 安全与配置提示
- GPU/CPU 切换前运行 `python -c "import jax; print(jax.default_backend())"` 验证后端；升级 CUDA 后更新 `requirements.txt` 中的备注。
- 密钥、硬件地址和云凭证通过环境变量或 `.env` 注入，禁止写入仓库；可用 `git ls-files | xargs rg -n "sk_"` 快速自查泄露。
- 大型日志、检查点和 `.mp4` 建议使用 Git LFS 或对象存储，只在仓库中保留复现所需的最小子集。
- 运行 `tensorboard` 或 Notebook 时务必绑定本地端口或启用密码，避免在共享 GPU 节点暴露控制权；关停后记得删除临时隧道。
