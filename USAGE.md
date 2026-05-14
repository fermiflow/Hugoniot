# 环境配置

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

# 网络训练

## 第一步：预训练原子核流模型

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

**预训练配置文件说明**

| 文件 | 粒子数 | Gamma 点 | GPU 配置 |
|------|--------|------|----------|
| `config14.yaml` | 14 | True | A800_80G x 1 |
| `twist14.yaml` | 14 | False | A800_80G x 1 |
| `config20.yaml` | 20 | True | A800_80G x 2 |
| `twist20.yaml` | 20 | False | A800_80G x 2 |
| `config32.yaml` | 32 | True | A800_80G x 4 |
| `twist32.yaml` | 32 | False | A800_80G x 4 |
| `config54.yaml` | 54 | True | A800_80G x 4 |
| `twist54.yaml` | 54 | False | A800_80G x 4 |

命名规则：
- `config{N}.yaml`：Gamma 点计算（kpt=[0,0,0]）
- `twist{N}.yaml`：twist-averaged 计算（kpt=[0.25,0.25,0.25]）

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

**预训练配置文件说明**

| 文件 | 粒子数 | Gamma 点 | GPU 配置 |
|------|--------|------|----------|
| `config14.yaml` | 14 | True | A800_80G x 2 |
| `twist14.yaml` | 14 | False | A800_80G x 2 |
| `config20.yaml` | 20 | True | A800_80G x 4 |
| `twist20.yaml` | 20 | False | A800_80G x 4 |
| `config32.yaml` | 32 | True | A800_80G x 8 |
| `twist32.yaml` | 32 | False | A800_80G x 8 |
| `config54.yaml` | 54 | True | A800_80G x 16 |
| `twist54.yaml` | 54 | False | A800_80G x 16 |

命名规则：
- `config{N}.yaml`：Gamma 点计算（kpt=[0,0,0]）
- `twist{N}.yaml`：twist-averaged 计算（kpt=[0.25,0.25,0.25]）

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
- **`acc_steps`**：梯度累积步数，等效批大小为 `batchsize × acc_steps`，每步实际占用显存不变，通过多次采样增加等效批大小。增大 `acc_steps` 可在不增加显存的前提下提升等效批大小，但训练时间会成倍增加。

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

# 参数含义

## 预训练参数

### 多机参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `num_hosts` | 参与训练的主机数量，单机时为 1 | 1 |

多机模式下还需额外指定 `server_addr` 和 `host_idx`。

### 体系参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `num` | 质子数（必须为偶数），决定模拟盒子大小 L=(4/3 pi num)^(1/3) | 32 |
| `dim` | 空间维度 | 3 |
| `rs` | Wigner-Seitz 半径（Bohr），控制密度 | 1.86 |
| `T` | 温度（Kelvin），内部转换为 Rydberg 单位：T/157888 | 10000 |
| `kpt` | k 点的分数坐标，[0,0,0] 为 Gamma 点 | [0.25, 0.25, 0.25] |

### PES 参数（`pes.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `pes.type` | 势能面类型，`hf`（Hartree-Fock）或 `dft` | hf |
| `pes.gamma` | 是否为 Gamma 点计算（kpt 非零时自动设为 False） | True |
| `pes.basis` | 基组名称 | gth-dzv |
| `pes.xc` | 交换关联泛函（DFT 时使用） | lda,vwn |
| `pes.rcut` | 实空间截断半径 | 24 |
| `pes.grid_length` | 实空间网格间距 | 0.5 |
| `pes.tol` | SCF 收敛阈值 | 1e-7 |
| `pes.max_cycle` | SCF 最大迭代次数 | 400 |
| `pes.use_jit` | 是否启用 JIT 编译加速 PES 计算 | True |
| `pes.Gmax` | 倒空间截断（平面波截断） | 15 |
| `pes.kappa` | Ewald 求和分割参数 | 10 |
| `pes.batchsize` | PES 计算的子批大小，需能整除 `batchsize` | 80 |

DIIS 加速参数（`pes.diis.*`）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `pes.diis.diis` | 是否启用 DIIS 加速 SCF 收敛 | True |
| `pes.diis.space` | DIIS 子空间大小 | 16 |
| `pes.diis.start_cycle` | 从第几步开始启用 DIIS | 6 |
| `pes.diis.damp` | DIIS 阻尼系数 | 0.2 |

Smearing 参数（`pes.smearing.*`）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `pes.smearing.smearing` | 是否启用电子占据数 smearing | True |
| `pes.smearing.method` | smearing 方法 | fermi |
| `pes.smearing.search.method` | 化学势搜索方法 | bisect |
| `pes.smearing.search.cycle` | 搜索最大迭代次数 | 300 |
| `pes.smearing.search.tol` | 搜索收敛阈值 | 1e-6 |

### Flow 网络参数（`flow.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `flow.steps` | flow 变换的堆叠次数 | 1 |
| `flow.depth` | FermiNet 网络深度（层数） | 6 |
| `flow.h1size` | 单粒子流隐藏层宽度 | 16 |
| `flow.h2size` | 双粒子流隐藏层宽度 | 16 |
| `flow.Nf` | 傅里叶特征数 | 5 |
| `flow.remat` | 是否启用梯度重计算（节省显存，增加计算时间） | False |

### 优化器参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `optimizer_flow` | 优化器类型：`sr`（随机重构）或 `cg`（共轭梯度） | sr |
| `lr_flow` | 学习率 | 1.0 |
| `decay_flow` | 学习率衰减系数 | 0.01 |
| `damping_flow` | Fisher 信息矩阵的阻尼项 | 0.001 |
| `maxnorm_flow` | 参数更新的最大范数裁剪 | 0.001 |
| `clip_factor` | 梯度裁剪因子 | 5.0 |
| `alpha` | SR 优化器的指数移动平均系数 | 0.1 |
| `gamma` | CG 优化器的 Fisher 矩阵混合系数 | 1 |

CG 优化器参数（`cg.*`，仅 `optimizer_flow=cg` 时生效）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `cg.mode` | Fisher 矩阵分解模式 | qr |
| `cg.init_vec_last_step` | 是否用上一步结果初始化 CG 向量 | False |
| `cg.solver.precondition` | 是否使用预条件 | False |
| `cg.solver.maxiter` | CG 求解器最大迭代次数 | None |
| `cg.solver.tol` | CG 求解器收敛阈值 | 1e-10 |
| `cg.solver.style` | 求解器类型 | cg |

### 训练参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `mc_therm` | MCMC 热化步数（从检查点续训时通常为 0） | 0 |
| `mc_width_p` | 质子 MCMC 提议步长（相对于盒子大小 L） | 0.05 |
| `mc_steps_p` | 每个训练步的 MCMC 采样步数 | 100 |
| `batchsize` | 每步训练的总样本数（必须能被 GPU 数整除） | 400 |
| `acc_steps` | 梯度累积步数，等效批大小 = batchsize x acc_steps | 2 |
| `epoch` | 总训练轮数 | 10000 |

### 其他参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `seed` | 随机数种子 | 42 |
| `folder` | 检查点和数据的保存根目录 | 见配置文件 |
| `load` | 续训时加载的检查点目录路径，None 表示从头训练 | None |
| `note` | 配置文件备注信息 | — |

## 训练参数

### 多机参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `num_hosts` | 参与训练的主机数量，单机时为 1 | 1 |

多机模式下还需额外指定 `server_addr` 和 `host_idx`。

### 体系参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `num` | 质子数（必须为偶数），决定模拟盒子大小 L=(4/3 pi num)^(1/3) | 32 |
| `dim` | 空间维度 | 3 |
| `rs` | Wigner-Seitz 半径（Bohr），控制密度 | 1.86 |
| `T` | 温度（Kelvin），内部转换为 Rydberg 单位：T/157888 | 10000 |
| `kpt` | k 点的分数坐标，[0,0,0] 为 Gamma 点 | [0.25, 0.25, 0.25] |

### LCAO 参数（`lcao.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `lcao.type` | 势能面类型，`hf`（Hartree-Fock）或 `dft` | hf |
| `lcao.gamma` | 是否为 Gamma 点计算（kpt 非零时自动设为 False） | False |
| `lcao.basis` | 基组名称 | gth-dzv |
| `lcao.xc` | 交换关联泛函（DFT 时使用） | lda,vwn |
| `lcao.rcut` | 实空间截断半径 | 24 |
| `lcao.grid_length` | 实空间网格间距 | 0.5 |
| `lcao.tol` | SCF 收敛阈值 | 1e-7 |
| `lcao.max_cycle` | SCF 最大迭代次数 | 300 |
| `lcao.use_jit` | 是否启用 JIT 编译加速 LCAO 计算 | True |
| `lcao.batchsize` | LCAO 计算的子批大小，需能整除 `batchsize` | 160 |

DIIS 加速参数（`lcao.diis.*`）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `lcao.diis.diis` | 是否启用 DIIS 加速 SCF 收敛 | True |
| `lcao.diis.space` | DIIS 子空间大小 | 16 |
| `lcao.diis.start_cycle` | 从第几步开始启用 DIIS | 6 |
| `lcao.diis.damp` | DIIS 阻尼系数 | 0.2 |

Smearing 参数（`lcao.smearing.*`）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `lcao.smearing.smearing` | 是否启用电子占据数 smearing | True |
| `lcao.smearing.method` | smearing 方法 | fermi |
| `lcao.smearing.search.method` | 化学势搜索方法 | bisect |
| `lcao.smearing.search.cycle` | 搜索最大迭代次数 | 400 |
| `lcao.smearing.search.tol` | 搜索收敛阈值 | 1e-6 |

### Flow 网络参数（`flow.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `flow.steps` | flow 变换的堆叠次数 | 1 |
| `flow.depth` | FermiNet 网络深度（层数） | 6 |
| `flow.h1size` | 单粒子流隐藏层宽度 | 16 |
| `flow.h2size` | 双粒子流隐藏层宽度 | 16 |
| `flow.Nf` | 傅里叶特征数 | 5 |
| `flow.remat` | 是否启用梯度重计算（节省显存，增加计算时间） | True |

### VAN 网络参数（`van.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `van.nlayers` | Transformer 层数 | 2 |
| `van.modelsize` | 模型嵌入维度 | 16 |
| `van.nheads` | 多头注意力头数 | 4 |
| `van.nhidden` | 前馈网络隐藏层宽度 | 32 |
| `van.remat` | 是否启用梯度重计算 | True |

### 波函数网络参数（`wfn.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `wfn.depth` | FermiNet 网络深度（层数） | 3 |
| `wfn.h1size` | 单粒子流隐藏层宽度 | 32 |
| `wfn.h2size` | 双粒子流隐藏层宽度 | 16 |
| `wfn.Nf` | 傅里叶特征数 | 5 |
| `wfn.remat` | 是否启用梯度重计算 | True |

### Ewald 参数（`ewald.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `ewald.Gmax` | 倒空间截断（平面波截断） | 15 |
| `ewald.kappa` | Ewald 求和分割参数 | 10 |

### 优化器参数（`optimizer.*`）

flow、van、wfn 三个模块各有独立的优化器配置，结构相同：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `optimizer.{module}.type` | 优化器类型：`sr`（随机重构）或 `cg`（共轭梯度） | sr |
| `optimizer.{module}.lr` | 学习率 | 0.3 / 1 |
| `optimizer.{module}.decay` | 学习率衰减系数 | 0.01 |
| `optimizer.{module}.damping` | Fisher 信息矩阵的阻尼项 | 0.001 |
| `optimizer.{module}.maxnorm` | 参数更新的最大范数裁剪 | 0.001 |
| `optimizer.{module}.clip_factor` | 梯度裁剪因子 | 5.0 |
| `optimizer.{module}.sr.alpha` | SR 优化器的指数移动平均系数 | 0.1 |

CG 优化器参数（`optimizer.{module}.cg.*`，仅 `type=cg` 时生效）：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `cg.mode` | Fisher 矩阵分解模式 | qr |
| `cg.gamma` | Fisher 矩阵混合系数 | 1 |
| `cg.init_vec_last_step` | 是否用上一步结果初始化 CG 向量 | False |
| `cg.solver.precondition` | 是否使用预条件 | False |
| `cg.solver.maxiter` | CG 求解器最大迭代次数 | None |
| `cg.solver.tol` | CG 求解器收敛阈值 | 1e-10 |
| `cg.solver.style` | 求解器类型 | cg |

### MCMC 参数（`mc.*`）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `mc.therm` | MCMC 热化步数（从检查点续训时通常为 0） | 10 |
| `mc.steps_p` | 每个训练步的质子 MCMC 采样步数 | 50 |
| `mc.steps_e` | 每个训练步的电子 MCMC 采样步数 | 1200 |
| `mc.width_p` | 质子 MCMC 提议步长（相对于盒子大小 L） | 0.04 |
| `mc.width_e` | 电子 MCMC 提议步长 | 0.06 |

### 训练参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `batchsize` | 每步训练的总样本数（必须能被 GPU 数整除） | 320 |
| `acc_steps` | 梯度累积步数，等效批大小 = batchsize × acc_steps | 4 |
| `epoch` | 总训练轮数 | 10000 |

### 其他参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `hutchinson` | 是否使用 Hutchinson 迹估计（节省显存，引入随机误差） | False |
| `seed` | 随机数种子 | 42 |
| `folder` | 检查点和数据的保存根目录 | 见配置文件 |
| `load` | 续训时加载的检查点目录路径，None 表示从头训练 | None |
| `load_pretrain.flow` | 预训练流检查点 `.pkl` 文件路径 | None |
| `load_pretrain.van` | 预训练 VAN 检查点 `.pkl` 文件路径 | None |
| `load_pretrain.wfn` | 预训练波函数检查点 `.pkl` 文件路径 | None |
| `note` | 配置文件备注信息 | — |


