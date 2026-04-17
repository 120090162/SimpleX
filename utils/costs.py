import crocoddyl
import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R


# default weight setting
DEFAULT_WQ = np.array([20.0, 20.0, 80.0] + [10.0] * 3 + [1.0] * 12)
DEFAULT_ALPHA = 30.0
DEFAULT_BETA = 10.0
DEFAULT_CU = 2e-4
DEFAULT_CF = 1.0
DEFAULT_CA = 2e3
DEFAULT_CS = 1e-2
DEFAULT_RHO = 2.0
DEFAULT_C1 = -30.0


def _cost_names(costs: crocoddyl.CostModelSum):
    """Return all registered cost names from a CostModelSum."""
    return costs.active_set.toset() | costs.inactive_set.toset()


def _replace_cost(
    costs: crocoddyl.CostModelSum,
    name: str,
    cost: crocoddyl.CostModelAbstract,
    weight=1.0,
):
    """Insert or overwrite a named cost term in ``costs``."""
    if name in _cost_names(costs):
        costs.removeCost(name)
    costs.addCost(name, cost, weight)


def make_state_weights(wq=DEFAULT_WQ, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """Build running/terminal state weights.

    Args:
        wq: Position/configuration weight vector.
        alpha: Velocity scaling factor, with ``wv = wq / alpha``.
        beta: Terminal scaling factor, with ``wx_terminal = beta * wx``.

    Returns:
        Tuple ``(wx, wx_terminal)`` used by weighted quadratic activations.
    """
    wq = np.asarray(wq, dtype=float)
    wv = wq / float(alpha)
    wx = np.concatenate([wq, wv])
    wx_terminal = float(beta) * wx
    return wx, wx_terminal


def make_reference_state(
    state: crocoddyl.StateMultibody,
    q_nominal: np.ndarray,
    base_position=None,
    base_rpy=None,
    v_ref=None,
):
    """Construct a reference state ``x_ref = [q_ref, v_ref]``.

    Args:
        state: Crocoddyl multibody state model.
        q_nominal: Nominal configuration used as base.
        base_position: Optional xyz override for floating-base position.
        base_rpy: Optional xyz Euler angles (rad) for floating-base orientation.
        v_ref: Optional generalized velocity reference.

    Returns:
        Full state reference vector of shape ``(state.nx,)``.
    """
    x_ref = np.concatenate([np.asarray(q_nominal).copy(), np.zeros(state.nv)])
    if base_position is not None:
        x_ref[:3] = np.asarray(base_position)
    if base_rpy is not None:
        quat_xyzw = R.from_euler("xyz", np.asarray(base_rpy)).as_quat()
        x_ref[3:7] = quat_xyzw
    if v_ref is not None:
        x_ref[state.nq :] = np.asarray(v_ref)
    return x_ref


def make_regulating_costs(
    state: crocoddyl.StateMultibody,
    actuation,
    x_ref: np.ndarray,
    wq=DEFAULT_WQ,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    cu=DEFAULT_CU,
):
    """Create regulating running/terminal costs.

    Args:
        state: Crocoddyl multibody state model.
        actuation: Crocoddyl actuation model (must expose ``nu``).
        x_ref: State reference used by the state residual.
        wq: Position/configuration weight vector.
        alpha: Velocity scaling factor.
        beta: Terminal state scaling factor.
        cu: Per-joint control regularization weight.

    Returns:
        ``(running_cost, terminal_cost)`` as two ``CostModelSum`` objects.
    """
    wx, wx_terminal = make_state_weights(wq=wq, alpha=alpha, beta=beta)
    wu = np.full(actuation.nu, float(cu))

    running = crocoddyl.CostModelSum(state, actuation.nu)
    terminal = crocoddyl.CostModelSum(state, actuation.nu)

    x_run = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModelWeightedQuad(2.0 * wx),
        crocoddyl.ResidualModelState(state, x_ref, actuation.nu),
    )
    x_term = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModelWeightedQuad(2.0 * wx_terminal),
        crocoddyl.ResidualModelState(state, x_ref, actuation.nu),
    )
    u_run = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModelWeightedQuad(2.0 * wu),
        crocoddyl.ResidualModelControl(state, actuation.nu),
    )

    running.addCost("lr_x", x_run, 1.0)
    running.addCost("lr_u", u_run, 1.0)
    terminal.addCost("lr_N", x_term, 1.0)
    return running, terminal


def sigmoid_weight(height: float, c1=DEFAULT_C1):
    """Compute the height gate ``S(c1*phi)`` used by foot slip/clearance."""
    return 1.0 / (1.0 + np.exp(-float(c1) * float(height)))


def set_foot_slip_clearance_cost(
    costs: crocoddyl.CostModelSum,
    state: crocoddyl.StateMultibody,
    actuation,
    frame_id: int,
    foot_height: float,
    cf=DEFAULT_CF,
    c1=DEFAULT_C1,
    cost_prefix="lf",
):
    """Set one foot slip-clearance cost term in a cost sum.

    Implements ``lf = cf * S(c1*phi) * ||v_t||^2`` where only tangential
    frame velocity components are weighted.

    Args:
        costs: Target ``CostModelSum``.
        state: Crocoddyl multibody state model.
        actuation: Crocoddyl actuation model.
        frame_id: Foot frame id.
        foot_height: Current foot height ``phi``.
        cf: Slip-clearance scalar weight.
        c1: Sigmoid steepness parameter.
        cost_prefix: Name prefix for registered cost term.
    """
    wt = sigmoid_weight(foot_height, c1=c1)
    # lf = cf * S(c1*phi) * ||v_t||^2, v_t: frame velocity x/y
    w_act = np.array([2.0 * cf * wt, 2.0 * cf * wt, 0.0, 0.0, 0.0, 0.0])
    residual = crocoddyl.ResidualModelFrameVelocity(
        state,
        frame_id,
        pinocchio.Motion.Zero(),
        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        actuation.nu,
    )
    lf_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(w_act), residual
    )
    _replace_cost(costs, f"{cost_prefix}_{frame_id}", lf_cost, 1.0)


def set_foot_slip_clearance_costs(
    costs: crocoddyl.CostModelSum,
    state: crocoddyl.StateMultibody,
    actuation,
    frame_ids,
    foot_heights,
    cf=DEFAULT_CF,
    c1=DEFAULT_C1,
    cost_prefix="lf",
):
    """Batch version of :func:`set_foot_slip_clearance_cost`.

    Args:
        costs: Target ``CostModelSum``.
        state: Crocoddyl multibody state model.
        actuation: Crocoddyl actuation model.
        frame_ids: Iterable of foot frame ids.
        foot_heights: Iterable of corresponding foot heights.
        cf: Slip-clearance scalar weight.
        c1: Sigmoid steepness parameter.
        cost_prefix: Name prefix for registered cost terms.
    """
    for frame_id, h in zip(frame_ids, foot_heights):
        set_foot_slip_clearance_cost(
            costs,
            state,
            actuation,
            int(frame_id),
            float(h),
            cf=cf,
            c1=c1,
            cost_prefix=cost_prefix,
        )


def set_air_time_cost(
    costs: crocoddyl.CostModelSum,
    state: crocoddyl.StateMultibody,
    actuation,
    frame_id: int,
    active=True,
    ca=DEFAULT_CA,
    cost_prefix="la",
):
    """Enable/disable one air-time cost term for a foot.

    Implements ``la = ca * phi^2`` by weighting only z-translation residual.

    Args:
        costs: Target ``CostModelSum``.
        state: Crocoddyl multibody state model.
        actuation: Crocoddyl actuation model.
        frame_id: Foot frame id.
        active: Whether to keep the cost active.
        ca: Air-time scalar weight.
        cost_prefix: Name prefix for registered cost term.
    """
    name = f"{cost_prefix}_{frame_id}"
    if not active:
        if name in _cost_names(costs):
            costs.removeCost(name)
        return

    # la = ca * phi^2, phi: foot height(z)
    w_act = np.array([0.0, 0.0, 2.0 * float(ca)])
    residual = crocoddyl.ResidualModelFrameTranslation(
        state, frame_id, np.zeros(3), actuation.nu
    )
    la_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(w_act), residual
    )
    _replace_cost(costs, name, la_cost, 1.0)


def make_C2_diagonal(nu=12):
    """Build ``C2`` for diagonal-leg pairing (trot-like symmetry)."""
    D = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C2 = np.zeros((4, nu))
    C2[0:2, 0:3] = D
    C2[0:2, 9:12] = -D
    C2[2:4, 3:6] = D
    C2[2:4, 6:9] = -D
    return C2


def make_C2_pace(nu=12):
    """Build ``C2`` for same-side-leg pairing (pace-like symmetry)."""
    D = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C2 = np.zeros((4, nu))
    C2[0:2, 0:3] = D
    C2[0:2, 6:9] = -D
    C2[2:4, 3:6] = D
    C2[2:4, 9:12] = -D
    return C2


def make_C2_bounding(nu=12):
    """Build ``C2`` for front/back pairing (bounding-like symmetry)."""
    D = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C2 = np.zeros((4, nu))
    C2[0:2, 0:3] = D
    C2[0:2, 3:6] = -D
    C2[2:4, 6:9] = D
    C2[2:4, 9:12] = -D
    return C2


class SymmetricControlCostData(crocoddyl.CostDataAbstract):
    """Data cache for :class:`SymmetricControlCostModel`."""

    def __init__(self, model, collector):
        """Initialize temporary Jacobian storage.

        Args:
            model: Symmetric control cost model.
            collector: Crocoddyl data collector.
        """
        super().__init__(model, collector)
        self.Ru_cache = np.zeros((model.C2.shape[0], model.nu))


class SymmetricControlCostModel(crocoddyl.CostModelAbstract):
    """
    ls(ui) = cs * ||C2 * ui||^2
    """

    def __init__(self, state, C2, cs, nu=12):
        """Create symmetric-control cost.

        Args:
            state: Crocoddyl multibody state model.
            C2: Mapping matrix from joint torques to paired-leg differences.
            cs: Scalar weight in ``ls``.
            nu: Control dimension.
        """
        C2 = np.asarray(C2, dtype=float)
        assert C2.shape == (4, nu)
        assert cs >= 0.0
        activation = crocoddyl.ActivationModelQuad(C2.shape[0])
        super().__init__(state, activation, nu)
        self.C2 = C2
        self.cs = float(cs)
        self.alpha = np.sqrt(2.0 * self.cs)

    def calc(self, data, x, u=None):
        """Compute scalar symmetric-control cost value."""
        if u is None:
            data.cost = 0.0
            data.residual.r[:] = 0.0
            return
        data.residual.r[:] = self.alpha * (self.C2 @ u)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        """Compute analytic derivatives of symmetric-control cost."""
        if u is None:
            data.Lx[:] = 0.0
            data.Lu[:] = 0.0
            data.Lxx[:, :] = 0.0
            data.Lxu[:, :] = 0.0
            data.Luu[:, :] = 0.0
            return
        data.Ru_cache[:, :] = self.alpha * self.C2
        self.activation.calcDiff(data.activation, data.residual.r)
        data.Lu[:] = data.Ru_cache.T @ data.activation.Ar
        data.Luu[:, :] = data.Ru_cache.T @ data.activation.Arr @ data.Ru_cache
        data.Lx[:] = 0.0
        data.Lxx[:, :] = 0.0
        data.Lxu[:, :] = 0.0

    def createData(self, collector):
        """Create model-specific cost data."""
        return SymmetricControlCostData(self, collector)


def add_symmetric_control_cost(
    costs: crocoddyl.CostModelSum,
    state: crocoddyl.StateMultibody,
    actuation,
    gait="trot",
    cs=DEFAULT_CS,
    cost_name="ls",
):
    """Add/update symmetric-control cost in a cost sum.

    Args:
        costs: Target ``CostModelSum``.
        state: Crocoddyl multibody state model.
        actuation: Crocoddyl actuation model.
        gait: One of ``trot``, ``pace``, or ``bounding``.
        cs: Scalar weight of symmetric-control term.
        cost_name: Registered name in the cost sum.
    """
    if gait == "trot":
        C2 = make_C2_diagonal(actuation.nu)
    elif gait == "pace":
        C2 = make_C2_pace(actuation.nu)
    elif gait == "bounding":
        C2 = make_C2_bounding(actuation.nu)
    else:
        raise ValueError(f"Unsupported gait for symmetric cost: {gait}")
    ls_cost = SymmetricControlCostModel(state, C2=C2, cs=cs, nu=actuation.nu)
    _replace_cost(costs, cost_name, ls_cost, 1.0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.params import _ASSETS_DIR
    from utils.sim_utils import find_quadruped_foot_frame_ids

    model_path = _ASSETS_DIR / "unitree_go2/go2.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found: {model_path}")

    pin_model = pinocchio.buildModelFromMJCF(model_path.as_posix())
    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    q_nominal = (
        pin_model.referenceConfigurations["home"]
        if "home" in pin_model.referenceConfigurations
        else pinocchio.neutral(pin_model)
    )
    x_ref = make_reference_state(
        state,
        q_nominal,
        base_position=q_nominal[:3] + np.array([0.2, 0.0, 0.0]),
        base_rpy=np.array([0.0, 0.0, 0.0]),
    )

    running_cost, terminal_cost = make_regulating_costs(state, actuation, x_ref)
    add_symmetric_control_cost(running_cost, state, actuation, gait="trot")

    pin_data = pin_model.createData()
    v0 = np.zeros(pin_model.nv)
    pinocchio.forwardKinematics(pin_model, pin_data, q_nominal, v0)
    pinocchio.updateFramePlacements(pin_model, pin_data)

    foot_frame_ids = find_quadruped_foot_frame_ids(pin_model)
    if foot_frame_ids:
        print(f"Found foot frames with ids: {foot_frame_ids}")
        heights = [pin_data.oMf[i].translation[2] for i in foot_frame_ids]
        set_foot_slip_clearance_costs(
            running_cost, state, actuation, foot_frame_ids, heights
        )
        set_air_time_cost(
            running_cost, state, actuation, foot_frame_ids[0], active=True
        )

    collector = crocoddyl.DataCollectorMultibody(pin_model.createData())
    pinocchio.forwardKinematics(pin_model, collector.pinocchio, q_nominal, v0)
    pinocchio.updateFramePlacements(pin_model, collector.pinocchio)

    x0 = np.concatenate([q_nominal, v0])
    u0 = np.zeros(actuation.nu)

    running_data = running_cost.createData(collector)
    running_cost.calc(running_data, x0, u0)
    running_cost.calcDiff(running_data, x0, u0)

    terminal_data = terminal_cost.createData(collector)
    terminal_cost.calc(terminal_data, x0)
    terminal_cost.calcDiff(terminal_data, x0)

    print("[costs.py] debug examples ready")
    print("active running costs:", sorted(list(_cost_names(running_cost))))
    print("running cost value:", running_data.cost)
    print("terminal cost value:", terminal_data.cost)
    print(
        "default weights:",
        {"cu": DEFAULT_CU, "cf": DEFAULT_CF, "ca": DEFAULT_CA, "cs": DEFAULT_CS},
    )
