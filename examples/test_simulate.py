import numpy as np
import pinocchio as pin

from absl import app
from absl import flags
from absl import logging

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cb_models import create_cubes, addCube
from utils.viz_utils import sub_sample
from pycontact.simulators import CCPADMMSimulator, CCPPGSSimulator, PinNCPADMMSimulator

# absl flags for command line arguments
FLAGS = flags.FLAGS
flags.DEFINE_bool("drag", False, "Enable drag torque")
flags.DEFINE_bool("display", False, "Enable display")
flags.DEFINE_bool("record", False, "Enable recording")
flags.DEFINE_bool("plot", False, "Enable plotting")
flags.DEFINE_integer("seed", 1234, "Random seed")
flags.DEFINE_bool("save", False, "Save results")
flags.DEFINE_bool("debug", False, "Enable debug mode")


def process_args():
    if FLAGS.record:
        FLAGS.display = True


import meshcat
import os
from pathlib import Path


def main(argv):
    del argv

    process_args()
    np.random.seed(FLAGS.seed)
    pin.seed(FLAGS.seed)

    a = 0.2  # size of cube
    m1 = 1e-3  # mass of cube 1
    m2 = 1e3  # mass of cube 2
    mu1 = 0.9  # friction parameter between cube and floor
    mu2 = 0.95  # friction parameter between the 2 cubes
    el = 0.0
    comp = 0.0
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = (
        create_cubes([a], [m1], mu1, el)
    )

    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addCube(
        model,
        geom_model,
        visual_model,
        actuation,
        a,
        m2,
        mu2,
        el,
        comp,
        color=np.array([1.0, 0.2, 0.2, 1.0]),
    )

    # Number of time steps
    T = 100
    # T = 1
    # time steps
    dt = 1e-3

    # Physical parameters of the contact problem
    Kb = 1e-4 * 0.0  # Baumgarte
    eps = 0.0  # elasticity

    # initial state
    q0 = pin.neutral(model)
    q0[2] = a / 2 + a / 50.0
    q0[9] = 3.0 * a / 2 + 3 * a / 50.0
    v0 = np.zeros(model.nv)
    q, v = q0.copy(), v0.copy()

    # simulator
    # simulator = CCPPGSSimulator(statistics = True)
    simulator = PinNCPADMMSimulator(warm_start=True)
    # simulator = CCPADMMSimulator(statistics=True)

    simulator.setSimulation(model, data, geom_model, geom_data)

    # record quantities during trajectory
    xs = [np.concatenate((q0, v0))]
    lams = []
    Js = []
    us = []
    Rs = []
    es = []
    Gs = []
    gs = []
    ncp_crits = []
    signorini_crits = []
    comp_crits = []
    if FLAGS.drag:

        def ext_torque(t):
            return np.min(
                [0.5 * (mu1 + mu2) * 9.81 * (m1 + m2), t * 4 * 9.81 * (m1 + m2)]
            )

    else:

        def ext_torque(t):
            return 0.0

    for t in range(T):
        tau_act = np.zeros(actuation.shape[1])
        tau = actuation @ tau_act
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        fext[1].linear[1] = ext_torque(t * dt)
        fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
        q, v = simulator.step(
            model,
            data,
            geom_model,
            geom_data,
            q,
            v,
            tau,
            fext,
            dt,
            Kb,
            100,
            1e-12,
            rel_th_stop=1e-12,  # , eps_reg = 0.
        )
        if FLAGS.debug:
            print("t=", t)
            simulator.plotCallback(block=False)
            input("Press enter for next step!")
        lams += [simulator.lam]
        Js += [simulator.J]
        Rs += [simulator.R]
        es += [simulator.signed_dist]
        Gs += [simulator.Del.G_]
        gs += [simulator.g]
        xs += [np.concatenate((q, v))]
        us += [tau]
        if len(simulator.solver.stats_.ncp_comp_) > 0:
            ncp_crits += [simulator.solver.stats_.ncp_comp_[-1]]
            comp_crits += [simulator.solver.stats_.comp_[-1]]
            signorini_crits += [simulator.solver.stats_.sig_comp_[-1]]
        else:
            ncp_crits += [np.nan]
            comp_crits += [np.nan]
            signorini_crits += [np.nan]

    if FLAGS.debug:
        simulator.close_plot_callback()

    dir_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    os.makedirs(dir_path / "logs", exist_ok=True)
    os.makedirs(dir_path / "logs/simulate/", exist_ok=True)

    if FLAGS.save:
        np.savez(
            dir_path / "logs/simulate/contact_problem.npz",
            G=simulator.Del.G_,
            g=simulator.g,
            mus=simulator.mus,
        )

    if FLAGS.plot:  # plotting quantities accross time_steps
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        steps = [t for t in range(T)]
        pen_err = []
        for ei in es:
            nci = len(ei) // 3
            pen_erri = 0.0
            for j in range(nci):
                pen_erri = max(pen_erri, ei[3 * j + 2])
            if nci == 0:
                pen_erri = np.nan
            pen_err += [pen_erri]

        pen_err = np.array(pen_err)
        plt.figure()
        plt.plot(steps, pen_err, marker="+")
        plt.xlabel("time step")
        plt.ylabel("Penetration error")
        if FLAGS.save:
            plt.savefig(dir_path / "logs/simulate/pen_error.pdf")
        plt.close()

        plt.figure()
        plt.plot(steps, ncp_crits, marker="+")
        plt.xlabel("time step")
        plt.ylabel("Contact complementarity")
        if FLAGS.save:
            plt.savefig(dir_path / "logs/simulate/ncp_comp.pdf")
        plt.close()

        plt.figure()
        plt.plot(steps, signorini_crits, marker="+")
        plt.xlabel("time step")
        plt.ylabel("Signorini complementarity")
        if FLAGS.save:
            plt.savefig(dir_path / "logs/simulate/sig_comp.pdf")
        plt.close()

        plt.figure()
        plt.plot(steps, comp_crits, marker="+")
        plt.xlabel("time step")
        plt.ylabel("Problem complementarity")
        if FLAGS.save:
            plt.savefig(dir_path / "logs/simulate/prob_comp.pdf")
        plt.close()

        if FLAGS.save:
            np.save(dir_path / "logs/simulate/pen_err.npy", pen_err)
            np.save(dir_path / "logs/simulate/sig_comp.npy", signorini_crits)
            np.save(dir_path / "logs/simulate/ncp_comp.npy", ncp_crits)
            print(np.max(ncp_crits))
            np.save(dir_path / "logs/simulate/prob_comp.npy", comp_crits)

    if FLAGS.display:
        from pinocchio.visualize import MeshcatVisualizer

        # visualize the trajectory
        vizer = MeshcatVisualizer(model, geom_model, visual_model)
        vizer.initViewer(open=True, loadModel=True)

        vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([10, 10, 0.1])))
        placement = np.eye(4)
        placement[:3, 3] = np.array([0, 0, -0.05])
        vizer.viewer["plane"].set_transform(placement)
        vizer.display(q0)

        numrep = 2
        cp = [0.8, 0.0, 0.2]
        # camera positions for visualization
        cps_ = [cp.copy() for _ in range(numrep)]
        rps_ = [np.zeros(3)] * numrep
        qs = [x[: model.nq] for x in xs]
        vs = [x[model.nq :] for x in xs]

        max_fps = 30.0
        fps = min([max_fps, 1.0 / dt])
        qs = sub_sample(qs, dt * T, fps)
        vs = sub_sample(vs, dt * T, fps)

        def get_callback(i: int):
            def _callback(t):
                # vizer.setCameraPosition(cps_[i])
                # vizer.setCameraTarget(rps_[i])
                vizer.viewer["/Cameras/default/position"].set_property(
                    "position", list(cps_[i])
                )
                vizer.viewer["/Cameras/default/lookAt"].set_property(
                    "position", list(rps_[i])
                )
                pin.forwardKinematics(model, vizer.data, qs[t], vs[t])
                # vizer.drawFrameVelocities(base_link_id)

            return _callback

        input("[Press enter]")

        if FLAGS.record:
            ctx = vizer.create_video_ctx(
                dir_path / "logs/simulate/simulation.mp4", fps=fps
            )
        else:
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            for i in range(numrep):
                vizer.play(qs, 1.0 / fps, get_callback(i))


if __name__ == "__main__":
    app.run(main)
