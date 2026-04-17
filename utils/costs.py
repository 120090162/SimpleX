import crocoddyl
import numpy as np

from line_profiler import profile
from .transforms import *


# 对称步态cost data
class SymmetricControlCostData(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        super().__init__(model, collector)
        self.Ru = np.zeros((model.nr, model.nu))  # dr/du


# 对称步态cost model
class SymmetricControlCostModel(crocoddyl.CostModelAbstract):
    """
    l_s(u) = c_s ||C2 u||^2
    """

    def __init__(self, state, C2, cs, nu=12):
        C2 = np.asarray(C2, dtype=float)
        assert C2.shape == (4, nu)

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
        data = SymmetricControlCostData(self, collector)
        return data
