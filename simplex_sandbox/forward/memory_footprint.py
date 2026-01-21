import pinocchio as pin
import numpy as np
from simple_sandbox.utils.sim_utils import (
    SimulationArgs,
    addFloor,
    addMaterialAndCompliance,
    addSystemCollisionPairs,
)

try:
    import mujoco

    has_mujoco = True
except:
    print("Mujoco not found")
    has_mujoco = False
import os

args = SimulationArgs().parse_args()
allowed_solvers = ["ADMM", "PGS"]
if args.contact_solver not in allowed_solvers:
    print(
        f"Error: unsupported simulator. Avalaible simulators: {allowed_solvers}. Exiting"
    )
    exit(1)
np.random.seed(args.seed)
pin.seed(args.seed)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"simple_sandbox/robots/go2/mjcf/scene.xml"

n = 100
if has_mujoco:
    models = []
    datas = []
    for _ in range(n):
        m = mujoco.MjModel.from_xml_path(model_path)
        m.opt.solver = 0
        d = mujoco.MjData(m)
        # print(d.ncon)
        models.append(m)
        datas.append(d)

    for i in range(n):
        m = models[i]
        d = datas[i]
        mujoco.mj_step(m, d)
        # print(d.ncon)

    input("PRESS ENTER TO SWITCH TO PINOCCHIO")

import pinocchio as pin
import numpy as np
import simple
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def createSimulator(
    model: pin.Model,
    geom_model: pin.GeometryModel,
    max_num_contacts: int = 4,
    tol: float = 1e-8,
    tol_rel: float = 1e-12,
    mu_prox: float = 1e-4,
    maxit: int = 1000,
    Kp: float = 0.0,
    Kd: float = 0.0,
):
    data = model.createData()
    geom_data = geom_model.createData()
    simulator = simple.Simulator(model, data, geom_model, geom_data)
    simulator.admm_constraint_solver_settings.absolute_precision = tol
    simulator.admm_constraint_solver_settings.relative_precision = tol_rel
    simulator.admm_constraint_solver_settings.max_iter = maxit
    simulator.admm_constraint_solver_settings.mu = mu_prox
    simulator.constraint_problem.setMaxNumberOfContactsPerCollisionPair(
        max_num_contacts
    )
    simulator.constraint_problem.Kp = Kp
    simulator.constraint_problem.Kd = Kd
    return simulator


model_path = f"simple_sandbox/robots/go2/mjcf/go2.xml"
model = pin.buildModelFromMJCF(model_path)
geom_model = pin.buildGeomFromMJCF(model, model_path, pin.COLLISION)
print("Done loading pinocchio model")

material = "concrete"
compliance = 0.0

visual_model = geom_model.copy()
addFloor(geom_model, visual_model)
addMaterialAndCompliance(geom_model, material, compliance)

q = pin.neutral(model)
q[2] = 0.5
v = np.zeros(model.nv)
addSystemCollisionPairs(model, geom_model, q)

print("Creating simple simulators...")
simulators = []
for _ in range(n):
    sim = createSimulator(model, geom_model)
    simulators.append(sim)
input("EOF")
