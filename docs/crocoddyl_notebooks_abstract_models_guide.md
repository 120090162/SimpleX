# Crocoddyl Notebooks 解析：自定义抽象模型给求解器使用

本文基于 `third_party/crocoddyl/notebooks` 的示例（重点是 `06_scaling_to_robotics.ipynb`、`arm_manipulation.ipynb`、`02_optimizing_a_cartpole_swingup.ipynb`）整理，给出如何编写并接入：

- `crocoddyl.DifferentialActionDataAbstract`
- `crocoddyl.DifferentialActionModelAbstract`
- `crocoddyl.CostModelAbstract`

并提供可直接改造的参考代码。

---

## 1. 从 notebooks 提炼出的关键模式

1. **DAM（微分动作模型）和 Data 必须成对出现**  
   在 `06_scaling_to_robotics.ipynb` 中，`DifferentialFreeFwdDynamicsModelDerived` 与 `DifferentialFreeFwdDynamicsDataDerived` 是成对设计的：  
   - `Model.calc / calcDiff` 负责动力学与代价导数  
   - `Data` 负责缓存 pinocchio data、actuation data、cost data

2. **CostModelAbstract 也必须有自己的 CostData**  
   `arm_manipulation.ipynb` 明确强调：自定义 cost 需要 data 类，而且应复用动力学阶段的 pinocchio data（通过 `DataCollectorMultibody` 传递）。

3. **给求解器使用时，DAM 通常包进 IntegratedActionModel**  
   使用 `crocoddyl.IntegratedActionModelEuler(dam, dt)` 形成离散模型，再组装 `ShootingProblem` 给 `SolverDDP/FDDP`。

4. **`calc` 与 `calcDiff` 必须保持一致的残差/导数语义**  
   `02_optimizing_a_cartpole_swingup.ipynb` 虽用的是 `ActionModelAbstract`，但同样展示了 Crocoddyl 对 `calc`/`calcDiff` 的一致性要求：  
   - `calc` 写入状态转移/代价  
   - `calcDiff` 写入一阶与二阶导（`Fx/Fu/Lx/Lu/Lxx/Luu/Lxu`）

---

## 2. 三个抽象类的最小实现要求

### 2.1 `crocoddyl.CostModelAbstract`

必须实现：

- `calc(self, data, x, u=None)`：写 `data.cost`（通常先写 `data.residual.r`）
- `calcDiff(self, data, x, u=None)`：至少写梯度（`Lx/Lu`），常见还写 Hessian（`Lxx/Luu/Lxu`）
- `createData(self, collector)`：返回自定义 `CostDataAbstract` 子类

常见模式（残差 + 激活函数）：

- `self.activation.calc(data.activation, data.residual.r)`
- `self.activation.calcDiff(data.activation, data.residual.r)`
- `data.Lx = Rx.T @ Ar`
- `data.Lxx = Rx.T @ Arr @ Rx`（Gauss-Newton 近似）

### 2.2 `crocoddyl.DifferentialActionModelAbstract`

必须实现：

- `calc(self, data, x, u=None)`：写连续时间动力学输出 `data.xout` 与 `data.cost`
- `calcDiff(self, data, x, u=None)`：写 `Fx/Fu` 与代价导数
- `createData(self)`：返回 `DifferentialActionDataAbstract` 子类

构造函数典型形式：

```python
crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr)
```

其中 `nr` 是 cost 残差维度（常由 `costModel.nr` 提供）。

### 2.3 `crocoddyl.DifferentialActionDataAbstract`

必须包含 DAM 计算所需缓存，常见包括：

- `pinocchio`：`pinocchio.Model.createData(model.state.pinocchio)`
- `multibody`：`crocoddyl.DataCollectorMultibody(self.pinocchio)`
- `actuation`：`model.actuation.createData()`
- `costs`：`model.costs.createData(self.multibody)`

并建议调用：

```python
self.costs.shareMemory(self)
```

用于复用共享缓存（notebooks 示例中采用该做法）。

---

## 3. 参考代码（可直接改造）

> 下面代码是“可用于求解器”的完整骨架：自定义 Cost + 自定义 DAM/Data + 组装 SolverFDDP。

```python
import numpy as np
import pinocchio
import crocoddyl


class FrameTranslationCostData(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        super().__init__(model, collector)
        self.J = np.zeros((3, model.state.nv))  # frame translation wrt q
        self.R = np.eye(3)


class FrameTranslationCostModel(crocoddyl.CostModelAbstract):
    def __init__(self, state, frame_id, target, nu):
        activation = crocoddyl.ActivationModelQuad(3)
        super().__init__(state, activation, nu)
        self.frame_id = frame_id
        self.target = np.asarray(target).reshape(3)

    def calc(self, data, x, u=None):
        # residual: p_frame - p_target
        data.residual.r[:] = (
            data.shared.pinocchio.oMf[self.frame_id].translation - self.target
        )
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.R[:, :] = data.shared.pinocchio.oMf[self.frame_id].rotation
        data.J[:, :] = data.R @ pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.shared.pinocchio,
            self.frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )[:3, :]

        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:, :] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = data.residual.Rx.T @ data.activation.Ar
        data.Lxx[:, :] = data.residual.Rx.T @ data.activation.Arr @ data.residual.Rx

    def createData(self, collector):
        return FrameTranslationCostData(self, collector)


class MyDifferentialActionData(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        super().__init__(model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)


class MyDifferentialActionModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuation_model, cost_model):
        super().__init__(state, actuation_model.nu, cost_model.nr)
        self.actuation = actuation_model
        self.costs = cost_model

    def calc(self, data, x, u=None):
        q = x[: self.state.nq]
        v = x[-self.state.nv :]

        if u is None:
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
            return

        self.actuation.calc(data.actuation, x, u)
        tau = data.actuation.tau
        data.xout[:] = pinocchio.aba(self.state.pinocchio, data.pinocchio, q, v, tau)

        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
            return

        q = x[: self.state.nq]
        v = x[-self.state.nv :]

        self.actuation.calcDiff(data.actuation, x, u)
        tau = data.actuation.tau

        pinocchio.computeABADerivatives(self.state.pinocchio, data.pinocchio, q, v, tau)
        ddq_dq = data.pinocchio.ddq_dq
        ddq_dv = data.pinocchio.ddq_dv

        data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.pinocchio.Minv @ data.actuation.dtau_dx
        data.Fu[:, :] = data.pinocchio.Minv @ data.actuation.dtau_du

        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return MyDifferentialActionData(self)
```

---

## 4. 将上述类接入 Crocoddyl 求解器

```python
# 假设你已有 robot_model, x0, frame_id, target
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)

track_cost = FrameTranslationCostModel(
    state=state,
    frame_id=frame_id,
    target=target,
    nu=actuation.nu,
)
xreg_cost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state, nu=actuation.nu))
ureg_cost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu=actuation.nu))

running_costs = crocoddyl.CostModelSum(state, nu=actuation.nu)
terminal_costs = crocoddyl.CostModelSum(state, nu=actuation.nu)
running_costs.addCost("track", track_cost, 1e2)
running_costs.addCost("xreg", xreg_cost, 1e-4)
running_costs.addCost("ureg", ureg_cost, 1e-6)
terminal_costs.addCost("track", track_cost, 1e4)
terminal_costs.addCost("xreg", xreg_cost, 1e-4)

dt = 1e-3
running_dam = MyDifferentialActionModel(state, actuation, running_costs)
terminal_dam = MyDifferentialActionModel(state, actuation, terminal_costs)
running_model = crocoddyl.IntegratedActionModelEuler(running_dam, dt)
terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_dam, 0.0)

T = 200
problem = crocoddyl.ShootingProblem(x0, [running_model] * T, terminal_model)
solver = crocoddyl.SolverFDDP(problem)

xs_init = [x0.copy() for _ in range(T + 1)]
us_init = [np.zeros(actuation.nu) for _ in range(T)]
solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)
```

---

## 5. 常见坑（基于 notebooks 实战）

1. `nu/nr` 维度不一致：`DAM.__init__(..., nu, nr)` 必须和 actuation/cost 对齐。  
2. 忘记在 `createData` 里建立 `DataCollectorMultibody`：会导致自定义 Cost 无法访问共享 pinocchio data。  
3. `calc` 和 `calcDiff` 不匹配：`calc` 用了什么残差定义，`calcDiff` 就要对应同一套 Jacobian/Hessian。  
4. 终端模型 `dt` 通常设为 `0.0`，并按需要只保留终端代价。  

---

## 6. 对应参考来源（本仓库内）

- `third_party/crocoddyl/notebooks/06_scaling_to_robotics.ipynb`  
  自定义 `DifferentialActionModelAbstract` / `DifferentialActionDataAbstract` 的标准模板。
- `third_party/crocoddyl/notebooks/arm_manipulation.ipynb`  
  关于 cost data 与 `DataCollectorMultibody` 的说明。
- `third_party/crocoddyl/unittest/bindings/factory.py`  
  多种 `CostModelAbstract` 派生实现（状态、控制、frame 位姿/位置/旋转等）。
- `third_party/crocoddyl/bindings/python/crocoddyl/utils/pendulum.py`  
  轻量级自定义 cost 模型示例。

---

## 7. `calcDiff` 中 `data.Fx` 与 `data.Fu` 的数学含义（补充）

在 `DifferentialActionModel` 里，`data.Fx` 和 `data.Fu` 是**连续时间输出 `xout` 对状态/控制的 Jacobian**。  
对于 `06_scaling_to_robotics.ipynb` 的例子，`xout = \ddot{q}`，因此

\[
x = (q, v), \quad u \in \mathbb{R}^{n_u}, \quad f(x,u)=\ddot q(q,v,u)\in\mathbb{R}^{n_v}
\]

\[
F_x=\frac{\partial f}{\partial x}
=\left[\frac{\partial \ddot q}{\partial q}\;\;\frac{\partial \ddot q}{\partial v}\right]
\in \mathbb{R}^{n_v\times n_{dx}}
\]

\[
F_u=\frac{\partial f}{\partial u}
=\frac{\partial \ddot q}{\partial u}
\in \mathbb{R}^{n_v\times n_u}
\]

在 ABA 分支中，
\[
\ddot q = M(q)^{-1}\big(\tau(x,u)-h(q,v)\big)
\]

代码对应的链式法则是：

\[
F_x=
\underbrace{
\left[\frac{\partial \ddot q}{\partial q}\;\;\frac{\partial \ddot q}{\partial v}\right]_{\tau\ \text{固定}}
}_{\text{Pinocchio: }ddq\_dq,\,ddq\_dv}
 + M^{-1}\frac{\partial \tau}{\partial x}
\]

\[
F_u= M^{-1}\frac{\partial \tau}{\partial u}
\]

对应实现：

```python
data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.pinocchio.Minv @ data.actuation.dtau_dx
data.Fu[:, :] = data.pinocchio.Minv @ data.actuation.dtau_du
```

注意：这里是**微分模型输出 `xout=\ddot q` 的导数**，不是离散一步 \(x_{k+1}\) 的导数。

---

## 8. `cimpc_sandbox/utils/models.py` 中 `DAM_contact.calcDiff` 的 \(F_x,F_u\) 公式

下面对应 `DAM_contact.calcDiff` 的两条分支（`data.real_collision` 为 False / True）。

记号：

- \(x=(q,v)\), \(u\) 为控制
- \(\tau=\tau(x,u)\), \(\tau_x=\frac{\partial\tau}{\partial x}\), \(\tau_u=\frac{\partial\tau}{\partial u}\)
- `xout` 对应 \(\ddot q\)
- \(M^{-1}=M(q)^{-1}\)

### 8.1 分支一：`not data.real_collision`

代码：

```python
data.Fx = [ddq_dq ddq_dv] + Minv @ dtau_dx
data.Fu = Minv @ dtau_du
```

数学上即
\[
f(x,u)=\ddot q(q,v,\tau)=M^{-1}(\tau-h)
\]
\[
F_x=\frac{\partial f}{\partial x}
=\left[\frac{\partial \ddot q}{\partial q}\;\;\frac{\partial \ddot q}{\partial v}\right]_{\tau\ \text{固定}}
+M^{-1}\tau_x
\]
\[
F_u=\frac{\partial f}{\partial u}=M^{-1}\tau_u
\]

这与标准 ABA 线性化一致。

### 8.2 分支二：`data.real_collision == True`

该分支可写成“基线 ABA + 接触冲量修正”：

1. **基线项**（先用 \(\tau+\text{effect}\) 调 `computeABADerivatives`）：
\[
F_x^{(0)}=
\left[\frac{\partial \ddot q}{\partial q}\;\;\frac{\partial \ddot q}{\partial v}\right]_{\tau+\text{effect}\ \text{固定}}
+M^{-1}\tau_x
\]
\[
F_u^{(0)}=M^{-1}\tau_u
\]

2. **接触修正项**（代码中的 `Fq,Fv,Ftau`）：
\[
\Delta F_x=\left[F_q\;\;F_v\right],\qquad
\Delta F_u=F_\tau\,\tau_u
\]
最终
\[
F_x=F_x^{(0)}+\Delta F_x,\qquad
F_u=F_u^{(0)}+\Delta F_u
\]

其中（完全对应代码）：
\[
F_q=\frac{1}{dt}M^{-1}\!\left(
\frac{\partial J^\top}{\partial q}\lambda
+J^\top\frac{\partial\lambda}{\partial q}
\right),
\]
\[
F_v=\frac{1}{dt}M^{-1}J^\top\frac{\partial\lambda}{\partial v},
\qquad
F_\tau=\frac{1}{dt}M^{-1}J^\top\frac{\partial\lambda}{\partial \tau}.
\]

这里 \(J\) 是代码中的 `contactJ`，\(\lambda\) 是接触冲量向量（`impulse`）。

冲量导数来自接触子问题线性化。代码里先构造
\[
A=J_{\text{left}}M^{-1}J_{\text{right}}^\top,\qquad
b=J_{\text{left}}(v+dt\,\ddot q_{\text{free}})
\]
并用 `contactAinv`（即 \((A+\mathrm{diag}(D))^{-1}\)）得到
\[
\frac{\partial\lambda}{\partial q},\;
\frac{\partial\lambda}{\partial v},\;
\frac{\partial\lambda}{\partial \tau}.
\]
若存在滑动接触，代码再用 \(E_s\)（`Es/Est`）把法向冲量导数映射为切向分量并拼接回完整 \(\lambda\) 导数。
