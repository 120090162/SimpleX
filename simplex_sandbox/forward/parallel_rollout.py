import pinocchio as pin
import numpy as np
import concurrent.futures
import timeit
import simple
import os

from simple_sandbox.utils.sim_utils import (
    addFloor,
    addMaterialAndCompliance,
)
from pin_utils import addSystemCollisionPairs

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


model = pin.buildModelFromMJCF(f"{current_dir}/robots/go2/mjcf/go2.xml")
geom_model = pin.buildGeomFromMJCF(
    model, f"{current_dir}/robots/go2/mjcf/go2.xml", pin.COLLISION
)

material = "concrete"
compliance = 0.0

visual_model = geom_model.copy()
addFloor(geom_model, visual_model)
addMaterialAndCompliance(geom_model, material, compliance)

q = pin.neutral(model)
q[2] = 0.5
v = np.zeros(model.nv)
addSystemCollisionPairs(model, geom_model, q)

batch_size = 1000
time_steps = 200
dt = 2e-3
# max_workers = 8
max_workers = os.cpu_count()  # from: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor

tau_batch = np.random.randn(batch_size, time_steps, model.nv) * 1.0
sim_batch = [createSimulator(model, geom_model) for _ in range(batch_size)]

# vizer = MeshcatVisualizer(model, geom_model, visual_model)
# vizer.initViewer(loadModel=True, open=True)
# vizer.display(q)


def simulate_tau_batch_sequential(tau_batch, sim_batch, q, v, dt):
    result_list = []
    for tau_traj, sim in zip(tau_batch, sim_batch):
        sim.reset()
        q_ = q.copy()
        v_ = v.copy()
        for tau in tau_traj:
            sim.step(q_, v_, tau, dt)
            q_ = sim.qnew
            v_ = sim.vnew
            # vizer.display(q_)
            # time.sleep(0.1)
        result_list.append((q_, v_))

    return result_list


def simulate_tau_batch_parallel(sim_batch, q, v, tau_batch, dt):
    def rollout_single(sim, tau):
        q_ = q.copy()
        v_ = v.copy()
        sim.rollout(q_, v_, tau, dt)  # rollout without GIL
        return sim.qnew, sim.vnew

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(rollout_single, sim, tau)
            for sim, tau in zip(sim_batch, tau_batch)
        ]
        results = [f.result() for f in futures]

    return results


# check if results match
results_test_sequential = simulate_tau_batch_sequential(
    tau_batch[:3], sim_batch[:3], q, v, dt
)
results_test_parallel = simulate_tau_batch_parallel(
    sim_batch[:3], q, v, tau_batch[:3, :3], dt
)

for i in range(3):
    assert np.allclose(results_test_sequential[i][0], results_test_parallel[i][0])
    assert np.allclose(results_test_sequential[i][1], results_test_parallel[i][1])

print(
    f"Running {batch_size} rollouts with {time_steps} time steps each and dt = {dt} seconds. Overall trajectory time = {time_steps * dt} seconds."
)
print(f"Number of workers: {max_workers}")

execution_time_sequential = timeit.timeit(
    stmt="simulate_tau_batch_sequential(tau_batch, sim_batch, q, v, dt)",
    setup="from __main__ import simulate_tau_batch_sequential, tau_batch, sim_batch, q, v, dt",
    number=10,
)

print(f"Execution time [sequential]: {execution_time_sequential:.6f} seconds")

execution_time_parallel = timeit.timeit(
    stmt="simulate_tau_batch_parallel(sim_batch, q, v, tau_batch, dt)",
    setup="from __main__ import simulate_tau_batch_parallel, sim_batch, q, v, tau_batch, dt",
    number=10,
)

print(f"Execution time [parallel]: {execution_time_parallel:.6f} seconds")
print(f"Speedup: {execution_time_sequential / execution_time_parallel:.2f}x")
