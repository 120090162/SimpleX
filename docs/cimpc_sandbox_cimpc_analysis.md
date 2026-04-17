# `cimpc_sandbox/utils/cimpc.py` 与论文实现细节对照（含修正建议）

本文基于以下材料交叉审视 `cimpc_sandbox/utils/cimpc.py`：

- `ref_doc/cost_design.pdf` 中 **Section 6: Implementation Detail for Motion Execution**
- `ref_doc/notes.pdf`（尤其 Notes 第 3 条）
- `cimpc_sandbox/utils/cimpc.py` 当前代码

目标：
1. 给出每个函数的实现逻辑与数学对应；
2. 判断与论文实现细节是否一致；
3. 提炼可复用的工程 trick；
4. 给出可直接迁移到新 `utils/costs.py` 与 `utils/models.py` 的参考代码骨架。

---

## 1. 论文中的“运动执行”关键点（Section 6 + Notes）

论文原文要点可归纳为：

1. **一拍移位 warm-start**  
   用上一次 MPC 解的
   \[
   (x_0,\dots,x_{N-1},u_0,\dots,u_{N-2})
   \]
   右移一格作为下一次初始化（Section 6）。

2. **最后一个控制输入设零（安全初始化）**  
   Notes 明确写到：不要直接复制 \(u_{N-2}\) 给 \(u_{N-1}\)，而是设 \(u_{N-1}=0\)，避免尾端大力矩导致末态发散。

3. **初始状态偏差 \( \tilde{x}_0 \neq x_0 \)**  
   实际估计状态 \( \tilde{x}_0 \) 往往不等于上轮预测首状态 \(x_0\)。  
   单次 shooting 很敏感，易触发接触模式级联变化并发散；多重 shooting/FDDP 对这类偏差更稳（Section 6）。

4. **FDDP 的价值**  
   接受状态-控制轨迹初始化并在早期迭代中逐步减小 dynamics gap，保持运动连续性。

5. **line search 与 rollout 成本高**  
   文中 profiling 指出 forward rollout/contact impulse 是主要时间开销之一（Figure 20）。

---

## 2. 当前 `cimpc.py` 是否符合论文细节

## 2.1 一致之处

### A) Shift warm-start：**一致**

各函数都在循环里做：
```python
xi = solver.xs.tolist()[1:]; xi.append(xi[-1])
ui = solver.us.tolist()[1:]; ui.append(np.zeros(12))
```
对应论文的一拍移位初始化。

### B) 末端控制置零：**一致**

`ui.append(np.zeros(12))` 与 Notes 第 3 条完全同向。

### C) 使用 FDDP：**一致**

使用 `crocoddyl.SolverBoxFDDP`，与文中采用 FDDP 方向一致。

---

## 2.2 不完全一致/可改进之处

### A) 初始状态应优先使用实时估计 \( \tilde{x}_0 \)

当前代码每轮重建问题时用：
```python
problem = ShootingProblem(xi[0], ...)
```
这是“用上轮预测首状态继续滚动”。  
论文 Section 6 强调真实执行时关键问题是 \( \tilde{x}_0 \neq x_0 \)，因此每轮应该显式注入当前估计状态 \( \tilde{x}_0 \)。

> 结论：离线/纯预测模拟可接受；真实机器人闭环时建议改为 `x_measured`。

### B) `np.zeros(12)` 写死维度

对 Go2 目前可用，但泛化到其他模型应改为：
```python
np.zeros(actuation.nu)
```
避免和新 `models.py`/`costs.py` 的通用接口冲突。

### C) `cimpc_adaptive` 的状态切片

当前：
```python
q, v = traj[-1][:state.nq], traj[-1][-state.nv:]
```
这版已经是正确切法（尾部取 `nv`），可跨 `nq != nv` 模型使用。  
若改成 `traj[-1][state.nv:]` 会错位，需避免。

### D) `flip` 函数存在覆盖赋值

```python
quat = rpy_to_quaternion([0, -pi, 0])
...
quat = rpy_to_quaternion([0, 0, 0])  # 覆盖了前一行
```
导致翻转姿态目标未真正生效；属于实验残留，不是论文公式本身。

### E) 缺少“反馈控制”落地步骤

`Feedback Control.pdf` 的 Section 6.3 在执行层明确给的是 **PD 跟踪器**。  
FDDP 的局部反馈增益 \(K\) 属于求解器内部策略（Crocoddyl 已实现），不要求在 `cimpc.py` 额外手写一层“\(u=u^\*-K\delta x\)”。

因此在 `cimpc.py` 侧应落地的是：
\[
p_{\text{target}} = \frac{u_0^\*}{K_p} + p_{\text{des}},
\]
\[
u_{\text{cmd}} = K_p(p_{\text{target}}-p) + K_d(\dot p_{\text{des}}-\dot p).
\]
等价写法：
\[
u_{\text{cmd}} = u_0^\* + K_p(p_{\text{des}}-p)+K_d(\dot p_{\text{des}}-\dot p).
\]
即“**FDDP 输出前馈力矩 + 关节 PD 修正**”。

另外，Section 6.2 的 gap contraction 也属于执行层关键公式，建议和反馈一起实现：
\[
\bar f_0 := \tilde x_0 \ominus x_0 \neq 0,\qquad
\bar f_{i+1} := f(x_i,u_i)\ominus x_{i+1}=0,\ \forall i.
\]
line-search 初始 rollout 状态：
\[
\hat x_0 := \tilde x_0 \oplus (\alpha-1)\bar f_0,
\]
并随迭代逐步缩小 dynamics gap。

Section 6.2 的 gap contraction 公式仍然是外循环初始化的重要依据：
\[
\bar f_0 := \tilde x_0 \ominus x_0 \neq 0,\qquad
\bar f_{i+1} := f(x_i,u_i)\ominus x_{i+1}=0,\ \forall i.
\]
\[
\hat x_0 := \tilde x_0 \oplus (\alpha-1)\bar f_0.
\]

**对应代码建议（可直接迁移）**：
```python
def compute_control_command(
    solver, x_meas,
    q_meas, v_meas,              # 当前关节位置/速度(仅受驱动关节)
    q_des, v_des,                # 来自下一时刻预测状态
    Kp, Kd,
):
    # 1) FDDP前馈
    u0_star = solver.us[0]

    # 2) Section 6.3 的 PD 执行层
    # ptarget = u*/Kp + q_des
    ptarget = u0_star / Kp + q_des
    u_cmd = Kp * (ptarget - q_meas) + Kd * (v_des - v_meas)
    # 等价: u_cmd = u0_star + Kp*(q_des-q_meas) + Kd*(v_des-v_meas)
    return u_cmd


def warmstart_with_gap_contraction(state, solver_prev, x_meas, alpha=1.0):
    # shift warm-start
    xs_init = solver_prev.xs[1:] + [solver_prev.xs[-1]]
    us_init = solver_prev.us[1:] + [np.zeros_like(solver_prev.us[0])]

    # fbar0 = x_meas ⊖ x0_prev
    fbar0 = state.diff(solver_prev.xs[0], x_meas)
    # xhat0 = x_meas ⊕ (alpha - 1) * fbar0
    xhat0 = state.integrate(x_meas, (alpha - 1.0) * fbar0)
    xs_init[0] = xhat0
    return xs_init, us_init
```

实现时注意：
- `q_des, v_des` 建议来自 `solver.xs[1]`（论文文字也是“next time step's joint state”）。  
- `q_meas, v_meas` 只取受驱动关节子空间（不是全状态）。  
- `Kp, Kd` 可以是标量或逐关节向量；若关节尺度差异大，建议向量化。  
- FDDP 内部反馈增益由求解器维护，`cimpc.py` 不需要重复实现同构反馈项。

---

## 3. 各函数逻辑、公式与工程技巧

## 3.1 `cimpc(total_time, x0, actionmodels, DT)`

### 逻辑
- 固定 `actionmodels`；
- 初始化后每周期迭代 1 次（`maxiter=1`）；
- shift warm-start + 末端零控制。

### 数学
\[
\bar x_i = x_{i+1}^\*,\quad
\bar u_i = u_{i+1}^\*,\quad
\bar u_{N-1}=0.
\]

### trick
- 实时性优先（RTI-like）；
- 重建 solver 简单可靠但有分配开销。

---

## 3.2 `cimpc_adaptive(...)`

### 逻辑
- 对每个接触点维护离地计数 `swing_count`；
- 当足高连续超过阈值后激活 air-time cost（`set_la_cost`）。

### 数学
令
\[
\delta_{k,i}=\mathbf{1}[h_{k,i}>\epsilon],\quad
a_k=\mathbf{1}[\text{连续离地计数}\ge i_t]
\]
则
\[
\ell_i = \ell_{\text{base},i} + \sum_k a_k\,c_a\,h_{k,i}^2.
\]

### trick
- 代价“事件触发”而非固定时序；
- 激活后保持，减小 gait 切换抖动。

---

## 3.3 `walking(...)`

### 逻辑
在线推进 \(x\) 方向目标：
\[
x^{ref}_{k+1} = x^{pred}_{0,k} + (v+0.1)\,DT\cdot20.
\]

### trick
- 用当前预测首状态更新目标，比纯时间函数更稳；
- 可注入 `u_static` 作为暖启动控制。

---

## 3.4 `flip(...)`

### 逻辑
- 分段升高 `z` 参考；
- 姿态目标写法有覆盖（当前不会真正 flip）。

### 建议
清理覆盖赋值并显式定义阶段目标（起跳/腾空/落地）。

---

## 3.5 `spining(...)`

### 逻辑
- yaw 参考离散积分：
\[
\psi_{k+1} = \psi_k + v\,DT\cdot20.
\]
- 四元数回写到 `xtarget[3:7]`。

### trick
- 直观易调；
- 注意欧拉角奇异区。

---

## 4. 与论文更一致的“执行层”参考代码（可迁移到新 costs/models）

下面给一个更接近 Section 6 的模板：

```python
def mpc_step(
    x_measured,              # 当前估计状态 x~0
    solver_prev,             # 上一轮 solver
    actionmodels,            # IAM_shoot(...) 结果
    state, actuation,
    maxiter=1, init_reg=0.1,
):
    problem = crocoddyl.ShootingProblem(
        x_measured, actionmodels[:-1], actionmodels[-1]
    )
    solver = crocoddyl.SolverBoxFDDP(problem)

    if solver_prev is None:
        xs_init = [x_measured] * (problem.T + 1)
        us_init = problem.quasiStatic([x_measured] * problem.T)
    else:
        xs_init = solver_prev.xs[1:] + [solver_prev.xs[-1]]
        us_init = solver_prev.us[1:] + [np.zeros(actuation.nu)]  # Notes: u_{N-1}=0

        # 关键：把首状态锚定到当前估计状态，体现 x~0 != x0
        xs_init[0] = x_measured.copy()

    solver.solve(xs_init, us_init, maxiter, False, init_reg)

    # 执行层使用 Section 6.3 的 PD（u* 为前馈）
    u_cmd = solver.us[0]

    return solver, u_cmd
```

这个骨架与论文要点一一对应：
- shift warm-start；
- 末端控制置零；
- 显式处理 \( \tilde{x}_0 \neq x_0 \)；
- 可接入反馈增益执行。

---

## 5. 与新 `utils/costs.py` / `utils/models.py` 的衔接建议

推荐把 `cimpc.py` 的“代价构建/更新”改成新接口：

1. 初始化使用 `make_regulating_costs(...)`；
2. gait 约束用 `add_symmetric_control_cost(...)`；
3. 脚部代价在线更新用 `set_foot_slip_clearance_costs(...)` 与 `set_air_time_cost(...)`；
4. 动力学模型序列用新 `utils/models.py` 的 `IAM_shoot(...)`。

这样可把“论文公式层”与“执行层 MPC 逻辑”清晰分离：  
`costs.py` 管数学代价定义，`models.py` 管动力学封装，`cimpc.py` 只管 MPC 外循环与 warm-start/执行策略。

---

## 6. 总结（审视结论）

`cimpc.py` 的核心 MPC trick（shift + `u_{N-1}=0` + FDDP）与论文 Section 6 主线是对齐的。  
主要差距在于真实执行时对 \( \tilde{x}_0 \) 的显式注入与反馈控制使用，这两点补上后会更贴近论文“Motion Execution”叙述，也更适合作为新 `costs.py`/`models.py` 的工程参考模板。
