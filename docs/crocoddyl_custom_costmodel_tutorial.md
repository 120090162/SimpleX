# Crocoddyl 自定义 `CostModelAbstract` 与 `CostData` 教程（基于 notebooks）

本文结合 `third_party/crocoddyl/notebooks` 的教程内容，说明如何实现一个自定义的：

- `crocoddyl.CostModelAbstract`
- `crocoddyl.CostDataAbstract`

并给出一个可直接用在四足机器人上的示例：**Symmetric Control Cost**（对称控制代价）。

---

## 1. notebooks 里对 cost 的关键信息

在 `notebooks/arm_manipulation.ipynb` 中，教程明确给出三点：

1. cost model 负责计算**标量代价**以及梯度/Hessian；
2. Crocoddyl 里常用“残差 + 激活函数（Gauss 近似）”形式；
3. 若自己写 cost，需要配套 cost data，并通过 `DataCollectorMultibody` 复用动力学阶段的 Pinocchio data。

相关片段可见：

- `arm_manipulation.ipynb` 中 “## II. Cost models” 章节；
- `dataCollector = crocoddyl.DataCollectorMultibody(robot.data)`；
- `trackData = goalTrackingCost.createData(dataCollector)`。

另外在 `06_scaling_to_robotics.ipynb` 中，`DAM` 的 data 里也是：

```python
self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
self.costs = model.costs.createData(self.multibody)
```

这说明自定义 cost data 应按同样方式接入。

---

## 2. 写一个自定义 `CostModelAbstract` 的最小步骤

### 2.1 Model 侧必须实现

- `__init__(...)`：调用 `crocoddyl.CostModelAbstract.__init__(...)`
- `calc(self, data, x, u=None)`：计算 `data.cost`（通常会先填 `data.residual.r`）
- `calcDiff(self, data, x, u=None)`：计算 `Lx, Lu, Lxx, Luu, Lxu` 中你需要的部分
- `createData(self, collector)`：返回你的 `CostDataAbstract` 子类

### 2.2 Data 侧必须实现

- 继承 `crocoddyl.CostDataAbstract`
- 在 `__init__` 调用父类构造
- 增加你自己的缓存（如 Jacobian、临时矩阵）

> 如果 cost 依赖 multibody 量（frame/com/Jacobian 等），`collector` 应为 `DataCollectorMultibody`，并通过 `data.shared.pinocchio` 访问共享数据。

---

## 3. 示例：Symmetric Control Cost（论文 5.1.5）

目标代价：
\[
l_s(u_i)=c_s\|C_2 u_i\|^2
\]

其中：

- \(u_i\in\mathbb{R}^{n_j}\), \(n_j=12\)
- \(C_2\in\mathbb{R}^{4\times 12}\)
- \(c_s>0\) 为标量权重

### 3.1 代码实现（自定义 cost model + data）

```python
import numpy as np
import crocoddyl


class CostDataSymmetricControl(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        super().__init__(model, collector)
        self.Ru = np.zeros((model.nr, model.nu))  # dr/du


class CostModelSymmetricControl(crocoddyl.CostModelAbstract):
    """
    l_s(u) = c_s ||C2 u||^2
    这里使用 ActivationModelQuad + 缩放残差:
        r = sqrt(2*c_s) * C2 u
    则 cost = 0.5 * ||r||^2 = c_s ||C2 u||^2
    """

    def __init__(self, state, C2, cs, nu=12):
        C2 = np.asarray(C2, dtype=float)
        assert C2.shape == (4, nu)
        assert cs >= 0.0

        activation = crocoddyl.ActivationModelQuad(C2.shape[0])  # nr=4
        super().__init__(state, activation, nu)

        self.C2 = C2
        self.cs = float(cs)
        self.alpha = np.sqrt(2.0 * self.cs)  # residual scaling

    def calc(self, data, x, u=None):
        if u is None:
            data.cost = 0.0
            data.residual.r[:] = 0.0
            return

        data.residual.r[:] = self.alpha * (self.C2 @ u)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lu[:] = 0.0
            data.Luu[:, :] = 0.0
            return

        data.Ru[:, :] = self.alpha * self.C2
        self.activation.calcDiff(data.activation, data.residual.r)

        # Lu = Ru^T * Ar
        data.Lu[:] = data.Ru.T @ data.activation.Ar

        # Luu = Ru^T * Arr * Ru
        data.Luu[:, :] = data.Ru.T @ data.activation.Arr @ data.Ru

        # 该 cost 与 x 无关
        data.Lx[:] = 0.0
        data.Lxx[:, :] = 0.0
        data.Lxu[:, :] = 0.0

    def createData(self, collector):
        return CostDataSymmetricControl(self, collector)
```

---

## 4. `C2` 的构造（Diagonal pairing / Pacing pairing）

先定义论文里的：
\[
D = [0_{2\times 1}, I_{2\times 2}] \in \mathbb{R}^{2\times 3}
\]

假设关节力矩顺序是按 4 条腿拼接，每条腿 3 维（如 `[ABD, HFE, KFE]`）：
\[
u=[u_{\text{leg0}},u_{\text{leg1}},u_{\text{leg2}},u_{\text{leg3}}]\in\mathbb{R}^{12}
\]

### 4.1 对角腿配对（trot 常见）

对应你给出的形式（示意）：
\[
C^{\text{diagonal}}_2=
\begin{bmatrix}
D & 0 & 0 & -D \\
0 & D & -D & 0
\end{bmatrix}
\]

### 4.2 同侧腿配对（pace 常见）

一种常见写法（在同样腿顺序假设下）：
\[
C^{\text{pace}}_2=
\begin{bmatrix}
D & 0 & -D & 0 \\
0 & D & 0 & -D
\end{bmatrix}
\]

> 注意：你的 `u` 里腿顺序若不同，只需要重排 block 列位置即可。

构造代码：

```python
import numpy as np

def make_C2_diagonal():
    D = np.hstack([np.zeros((2, 1)), np.eye(2)])  # [0, I]
    C2 = np.zeros((4, 12))
    C2[0:2, 0:3] = D
    C2[0:2, 9:12] = -D
    C2[2:4, 3:6] = D
    C2[2:4, 6:9] = -D
    return C2

def make_C2_pace():
    D = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C2 = np.zeros((4, 12))
    C2[0:2, 0:3] = D
    C2[0:2, 6:9] = -D
    C2[2:4, 3:6] = D
    C2[2:4, 9:12] = -D
    return C2
```

---

## 5. 如何接入到 Crocoddyl 求解

```python
# 1) 构造 cost
C2 = make_C2_diagonal()    # 或 make_C2_pace()
cs = 5e-3
sym_cost = CostModelSymmetricControl(state, C2=C2, cs=cs, nu=actuation.nu)

# 2) 加入 runningCostModel
runningCostModel.addCost("sym_ctrl", sym_cost, 1.0)

# 3) terminal 是否启用按需求
# terminalCostModel.addCost("sym_ctrl", sym_cost, 1.0)
```

这里 `1.0` 只是 `CostModelSum` 的外部权重；你已经在 `cs` 中编码了论文权重，不建议再重复放大。

---

## 6. 维度与导数核对清单

实现后建议先检查以下维度：

- `data.residual.r`: `(4,)`
- `data.Lu`: `(12,)`
- `data.Luu`: `(12, 12)`
- `data.Lx`: `(state.ndx,)` 且应为 0（该 cost 不依赖 `x`）
- `data.Lxu`: `(state.ndx, 12)` 且应为 0

并确认解析导数：
\[
\frac{\partial l_s}{\partial u}=2c_s C_2^\top C_2 u,\qquad
\frac{\partial^2 l_s}{\partial u^2}=2c_s C_2^\top C_2
\]
与实现一致。

---

## 7. 与 `CoMPositionCostModelDerived` 对照检查（factory.py）

`third_party/crocoddyl/unittest/bindings/factory.py` 中 `CoMPositionCostModelDerived` 的核心写法是：

1. `calc`：残差取
\[
r = \mathrm{com}(q)-c_{\mathrm{ref}} \in \mathbb{R}^3
\]
然后调用 activation 得到 `data.cost`。

2. `calcDiff`：构造
\[
R_x=\begin{bmatrix}J_{\mathrm{com}} & 0\end{bmatrix}
\]
并用
\[
L_x=R_x^\top A_r,\qquad
L_{xx}=R_x^\top A_{rr}R_x
\]
且该实现里 \(L_u,L_{uu},L_{xu}=0\)（因为此 cost 不显式依赖 \(u\)）。

3. 该实现通过 `data.shared.pinocchio.com[0]` 和 `Jcom` 取值，意味着在进入 cost 计算前，外层动力学/状态流程应已更新对应 Pinocchio 数据（例如 `computeAllTerms` 或等效的 CoM/Jacobian 更新链路）。

这与本文“自定义 cost model + cost data”的结构和导数写法是一致的。

---

## 8. `ActivationModelQuad` 的正确含义

`ActivationModelQuad(nr)` 在 Crocoddyl 中定义为标准二次激活：

\[
a(r)=\frac{1}{2}r^\top r
\]

其导数为：

\[
A_r=\frac{\partial a}{\partial r}=r,\qquad
A_{rr}=\frac{\partial^2 a}{\partial r^2}=I
\]

这与 `include/crocoddyl/core/activations/quadratic.hpp` 的实现完全一致（`a_value = 0.5*r.dot(r)`, `Ar = r`, `Arr = I`）。

因此本文示例中使用
\[
r=\sqrt{2c_s}\,C_2u
\]
时，确实得到
\[
a(r)=\frac12\|r\|^2=c_s\|C_2u\|^2
\]
与论文代价定义一致。
