# `utils/costs.py` 实现思路与数学公式对照

本文把当前 `utils/costs.py` 的核心函数与类逐一映射到数学表达，说明实现思路，并重点解释为什么 `SymmetricControlCost` 需要自定义类。

---

## 1. 统一约定

Crocoddyl 中常见代价写法为：

\[
\ell(r)=\frac{1}{2} r^\top W r
\]

其中 `ActivationModelWeightedQuad(w)` 对应对角权重矩阵 \(W=\mathrm{diag}(w)\)。

因此若目标想写成
\[
c \|z\|^2
\]
可以通过设置残差 \(r=z\) 且权重 \(W=2cI\) 来实现，因为
\[
\frac{1}{2} z^\top (2cI) z = c\|z\|^2.
\]

---

## 2. 权重构造相关函数

## 2.1 `make_state_weights(wq, alpha, beta)`

代码逻辑：
- \(w_v = \frac{w_q}{\alpha}\)
- \(w_x = [w_q,\; w_v]\)
- \(w_x^N = \beta\,w_x\)

对应文档里的权重设计（位置/速度分离、终端放大）。

## 2.2 `make_reference_state(...)`

构造
\[
x_\mathrm{ref} = [q_\mathrm{ref}, v_\mathrm{ref}]
\]
其中：
- 可覆盖 base 平移 \(q_{0:3}\)
- 可用 RPY 生成四元数覆盖 base 姿态 \(q_{3:7}\)
- 速度默认零或用户输入

---

## 3. Regulating cost（`make_regulating_costs`）

### 3.1 运行阶段状态项 `lr_x`

残差：
\[
r_x = x \ominus x_\mathrm{ref}
\]
权重取 \(2w_x\)，故
\[
\ell_{r,x} = \frac{1}{2} r_x^\top \mathrm{diag}(2w_x) r_x
= \|r_x\|_{w_x}^2.
\]

### 3.2 运行阶段控制项 `lr_u`

残差：
\[
r_u = u
\]
权重取 \(2w_u\)（其中 \(w_u = c_u \mathbf{1}\)），故
\[
\ell_{r,u} = \|u\|_{w_u}^2.
\]

### 3.3 终端项 `lr_N`

\[
\ell_{N} = \|x\ominus x_\mathrm{ref}\|_{w_x^N}^2,\quad w_x^N=\beta w_x.
\]

---

## 4. Foot slip + clearance（`set_foot_slip_clearance_cost`）

文中目标形式：
\[
\ell_f = c_f\,S(c_1\phi)\,\|v_t\|^2,\qquad
S(z)=\frac{1}{1+e^{-z}}
\]
其中 \(\phi\) 为足端高度，\(v_t\) 为切向速度（x/y）。

代码做法：
- 先算 \(w_t=S(c_1\phi)\)
- 使用 `ResidualModelFrameVelocity`，仅给速度前两维权重
- 权重向量设为 \([2c_fw_t,\;2c_fw_t,\;0,0,0,0]\)

因此代价是
\[
\ell_f
=\frac{1}{2}(2c_fw_t v_x^2 + 2c_fw_t v_y^2)
=c_fw_t(v_x^2+v_y^2),
\]
与目标一致。

批量版本 `set_foot_slip_clearance_costs` 只是逐足调用同一公式。

---

## 5. Air-time cost（`set_air_time_cost`）

文中目标形式：
\[
\ell_a = c_a \phi^2
\]

代码做法：
- 使用 `ResidualModelFrameTranslation`（残差是足端平移）
- 仅 z 方向权重非零，设为 \(2c_a\)

得到
\[
\ell_a = \frac{1}{2}(2c_a z^2)=c_a z^2.
\]
当 z 轴就是高度 \(\phi\) 时，严格对应 \(\ell_a=c_a\phi^2\)。

---

## 6. Symmetric control 矩阵（`make_C2_*`）

定义
\[
D=[0_{2\times 1}, I_{2\times 2}] \in \mathbb{R}^{2\times 3}
\]

`make_C2_diagonal / pace / bounding` 只是把 \(\pm D\) 放到不同腿的 block 上，构成
\[
C_2 \in \mathbb{R}^{4\times 12}
\]
用于提取并比较配对腿的 HFE/KFE 力矩差。

---

## 7. 为什么 `SymmetricControlCost` 要自定义类？

目标是：
\[
\ell_s(u)=c_s\|C_2u\|^2.
\]

`CostModelResidual + ResidualModelControl` 只能直接表示 \(u-u_\mathrm{ref}\) 一类残差；这里需要线性映射 \(C_2u\)，因此需要自定义 cost（或自定义 residual）。当前实现选择自定义 cost 类，最小改动地复用 Crocoddyl 的激活模型与数据结构。

### 7.1 当前实现

在 `SymmetricControlCostModel` 中定义残差：
\[
r(u)=\alpha C_2u,\quad \alpha=\sqrt{2c_s}.
\]
并使用 `ActivationModelQuad`：
\[
a(r)=\frac{1}{2}r^\top r.
\]
于是
\[
\ell_s = a(r)=\frac{1}{2}\|\alpha C_2u\|^2
=\frac{1}{2}(2c_s)\|C_2u\|^2
=c_s\|C_2u\|^2.
\]
与目标公式严格一致。

### 7.2 解析导数

\[
\frac{\partial \ell_s}{\partial u}
=2c_s C_2^\top C_2u,\qquad
\frac{\partial^2 \ell_s}{\partial u^2}
=2c_s C_2^\top C_2.
\]

代码里通过
- `Ar`, `Arr`（activation 的一阶/二阶导）
- `Ru_cache = \partial r/\partial u = \alpha C_2`

计算：
\[
L_u = R_u^\top A_r,\qquad
L_{uu}=R_u^\top A_{rr}R_u
\]
与上式一致。

---

## 8. 关于“只用了 Crocoddyl 已有组件，数学是否正确”

结论：**当前实现在数学上是自洽且正确的**，理由如下：

1. 每个 cost 都遵循同一二次激活定义 \(\frac12 r^\top W r\)；  
2. `2*weight` 的写法精确实现目标里的 \(c\| \cdot \|^2\) 形式；  
3. Symmetric cost 通过缩放残差 \(r=\sqrt{2c_s}C_2u\) 与 `ActivationModelQuad` 的组合，严格还原 \(\ell_s\)；  
4. 梯度与 Hessian 的实现与解析表达一一对应。

---

## 9. 备注（实现层面）

- `SymmetricControlCostData` 使用 `Ru_cache` 而不是 `data.Ru`，是因为当前 Python 绑定里 `data.Ru` 走向 deprecated/只读路径；`Ru_cache` 仅作为本地缓存，不改变数学意义。  
- `DEFAULT_RHO` 目前仅保留为参数常量，未在本文件 cost 计算中直接使用。  

