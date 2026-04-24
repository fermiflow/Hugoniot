# 使用指南

## 环境配置

克隆仓库：

```bash
git clone https://github.com/fermiflow/Hugoniot.git
cd Hugoniot
```

安装依赖：

```bash
pip install -r requirements.txt
```

> `hqc` 和 `cfgmanager` 已包含在仓库中，无需单独安装。

---

## 第一步：预训练核子流模型

运行 `pretrainflow.py`，对质子位置的归一化流进行预训练。

**基本用法**（使用默认配置 `conf/pretrain/flow/config.yaml`）：

```bash
python pretrainflow.py
```

**通过命令行覆盖参数：**

```bash
python pretrainflow.py num=16 rs=1.86 T=10000 batchsize=256
```

**关键参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num` | 质子数（必须为偶数） | 14 |
| `rs` | Wigner-Seitz 半径 | 2 |
| `T` | 温度（开尔文） | 20000 |
| `batchsize` | 批大小（必须能被 GPU 数整除） | 1024 |
| `epoch` | 训练轮数 | 10000 |
| `folder` | 检查点和数据的保存目录 | 见配置文件 |
| `load` | 续训时加载的检查点目录路径 | None |

**切换配置文件**：修改 `pretrainflow.py` 第 31 行：

```python
@hydra.main(version_base=None, config_path="conf/pretrain/flow", config_name="config")
```

将 `config_name` 改为 `conf/pretrain/flow/` 下任意 yaml 文件名（如 `config32`、`twist32`）。

**输出文件**（保存在 `folder` 目录下）：

- `epoch_XXXXXX.pkl` — 检查点文件（每 10 分钟自动保存）
- `data.txt` / `data.csv` — 每轮训练的物理量
- `config.yaml` — 保存的配置
- `alog.log` — 完整训练日志

---

## 第二步：主训练

运行 `main.py`，联合训练流模型、VAN（变分自回归网络）和波函数。

**基本用法**（使用默认配置 `conf/train/config.yaml`）：

```bash
python main.py
```

**通过命令行覆盖参数：**

```bash
python main.py num=16 rs=1.86 T=10000 batchsize=256
```

**加载预训练流的检查点：**

```bash
python main.py \
    num=16 \
    rs=1.86 \
    T=10000 \
    batchsize=256 \
    load_pretrain.flow=/path/to/pretrain/epoch_001000.pkl
```

**从检查点续训：**

```bash
python main.py load=/path/to/checkpoint/directory
```

**关键参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num` | 质子数（必须为偶数） | 14 |
| `rs` | Wigner-Seitz 半径 | 2 |
| `T` | 温度（开尔文） | 10000 |
| `batchsize` | 批大小（必须能被 GPU 数整除） | 256 |
| `epoch` | 训练轮数 | 10000 |
| `folder` | 检查点和数据的保存目录 | 见配置文件 |
| `load` | 续训时加载的检查点目录路径 | None |
| `load_pretrain.flow` | 预训练流检查点 `.pkl` 文件路径 | None |

**切换配置文件**：修改 `main.py` 第 36 行：

```python
@hydra.main(version_base=None, config_path="conf/train", config_name="config")
```

将 `config_name` 改为 `conf/train/` 下任意 yaml 文件名（如 `config32`、`twist32`）。

**输出文件**（保存在 `folder` 目录下）：

- `epoch_XXXXXX.pkl` — 检查点文件（每 10 分钟自动保存）
- `data.txt` — 每轮训练的物理量（自由能、能量、压强、熵、接受率）
- `config.yaml` — 保存的配置
- `alog.log` — 完整训练日志

---

## 典型工作流

```bash
# 第一步：对 rs=1.86、T=10000K、16 个原子进行预训练
python pretrainflow.py num=16 rs=1.86 T=10000 batchsize=256 \
    folder=/your/output/pretrain/

# 第二步：主训练，加载预训练流
python main.py num=16 rs=1.86 T=10000 batchsize=256 \
    load_pretrain.flow=/your/output/pretrain/n_16_rs_1.86_T_10000_.../epoch_001000.pkl \
    folder=/your/output/train/
```

---

## 参考配置：twist32 算例

`conf/pretrain/flow/twist32.yaml` 是在 **4×A800 80G** 上运行预训练的参数配置；
`conf/train/twist32.yaml` 是在 **8×A800 80G** 上运行主训练的参数配置，可通过 `load_pretrain.flow` 加载预训练结果。

两个配置的关键参数如下：

| 参数 | pretrain (twist32) | train (twist32) |
|------|--------------------|-----------------|
| `num` | 32 | 32 |
| `batchsize` | 400 | 320 |
| `pes/lcao.batchsize` | 80 | 160 |
| `acc_steps` | 2 | 4 |

### 各参数对显存的影响

- **`num`**：体系粒子数，是显存占用最主要的因素。`num` 越大，神经网络输入维度和 Slater 行列式规模都线性增长，显存需求显著上升。
- **`batchsize`**：每步训练使用的样本数，直接决定显存占用。`batchsize` 必须能被 GPU 总数整除，实际每卡的样本数为 `batchsize / num_gpus`。
- **`pes.batchsize` / `lcao.batchsize`**：调用 PES/LCAO 计算时的子批大小。该计算涉及量子化学求解，显存开销较大；减小此值可降低峰值显存，但会增加计算时间。需满足 `batchsize` 能被其整除。
- **`acc_steps`**：梯度累积步数，等效批大小为 `batchsize × acc_steps`，但每步实际占用显存不变。增大 `acc_steps` 可在不增加显存的前提下提升等效批大小。

### 建议：先用小体系测试

**首次运行时，建议先用 `num=4` 的小体系验证环境和配置是否正常**，确认能跑通后再切换到大体系：

```bash
# 用最小体系快速验证
python pretrainflow.py num=4 rs=1.86 T=10000 batchsize=16 epoch=10 \
    folder=/tmp/test_pretrain/

python main.py num=4 rs=1.86 T=10000 batchsize=16 epoch=10 \
    load_pretrain.flow=/tmp/test_pretrain/.../epoch_000010.pkl \
    folder=/tmp/test_train/
```

---

## 第三步：监控训练过程

训练过程中可以用 `src/inference/plot_data.py` 中的 `plot_data` 函数实时查看训练曲线：

```python
from src.inference.plot_data import plot_data

files = ["/your/output/train/n_32_.../data.txt"]
plot_data(files, quantities=['f', 'etot', 'p', 's'], running_average=10)
```

支持的 `quantities`：`'f'`（自由能）、`'etot'`（总能量）、`'p'`（压强）、`'k'`（动能）、`'s'`（熵）、`'se'`、`'sp'`、`'acc_s'`、`'acc_x'`（接受率）。

---

## 第四步：网络推理（采样）

训练收敛后，用 `src/inference/sample_x.py` 中的 `sample_sx` 函数对训练好的检查点做推理，获得更多独立样本以降低统计误差：

```python
from src.inference.sample_x import sample_sx

sample_sx(
    files=["/your/output/train/n_32_.../epoch_005000.pkl"],
    sample_total_batch=1024,  # 总采样数
    sample_batch=256,         # 每步采样数（受显存限制）
    sample_therm=10,          # 热化步数
)
```

采样结果会自动保存为 `epoch_XXXXXX_sample_sx_bs_1024.pkl`，供后续分析使用。

---

## 第五步：计算状态方程（EOS）

采样完成后，用 `src/inference/quantity.py` 中的 `eos` 函数计算并打印精确的状态方程：

```python
from src.inference.quantity import eos

eos(
    ckpt_files=["/your/output/train/n_32_.../epoch_005000.pkl"],
    isotope='H',   # 'H' 或 'D'（氘）
    unit='Ry',     # 'Ry', 'Ha', 'eV'
)
```

输出包括自由能 F、总能量 E、压强 P、熵 S 等，可通过 `save_csv_filename` 保存为 CSV 文件。

---

## 注意事项

- `batchsize` 必须能被 GPU 总数整除。
- 若不提供预训练流（`load_pretrain.flow=None`），流模型将从均匀分布初始化。
- 配置文件使用 [Hydra](https://hydra.cc/) 管理，所有参数均可在命令行中覆盖。
- 训练过程中每 10 分钟自动保存一次检查点。
