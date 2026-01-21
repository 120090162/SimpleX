import pinocchio as pin
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description as plr
from simple_sandbox.utils.sim_utils import (
    addRobotName,
    mergeModels,
    createFreeflyerModel,
    create3DPyramid,
    runSimpleSimulationFromModel,
    runMujocoXML,
    SimulationArgs,
)


class ScriptArgs(SimulationArgs):
    add_robots: bool = False
    add_g1: bool = False
    add_talos: bool = False
    add_go2: bool = False
    add_anymal: bool = False
    collision_pairs_init_margin: float = 1e-3  # override the default


args = ScriptArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)

if args.backend == "simple":
    # Create model
    # --> pyramid
    pyramid_model, pyramid_geom_model = create3DPyramid(
        1.0, 4, cube_size=0.25, spacing=1e-2
    )
    pyramid_visual_model = pyramid_geom_model.copy()
    addRobotName(pyramid_model, pyramid_geom_model, pyramid_visual_model, "pyramid")
    pyramid_q0 = pin.neutral(pyramid_model)
    pyramid_v0 = np.zeros(pyramid_model.nv)
    model = pyramid_model.copy()
    geom_model = pyramid_geom_model.copy()
    visual_model = pyramid_visual_model.copy()
    q0 = pyramid_q0.copy()
    v0 = pyramid_v0.copy()

    # --> g1
    if args.add_robots or args.add_g1:
        ff_model, ff_geom_model, ff_visual_model, ff_q0 = createFreeflyerModel(
            "g1_freeflyer"
        )
        g1_robot = plr("g1_description")
        g1_model, g1_geom_model, g1_visual_model = (
            g1_robot.model,
            g1_robot.collision_model,
            g1_robot.visual_model,
        )
        g1_q0 = pin.neutral(g1_model)
        g1_v0 = np.zeros(g1_model.nv)
        addRobotName(g1_model, g1_geom_model, g1_visual_model, "g1")
        g1_model, g1_geom_model, g1_visual_model, g1_q0, g1_v0 = mergeModels(
            ff_model,
            ff_geom_model,
            ff_visual_model,
            ff_q0,
            np.zeros(ff_model.nv),
            g1_model,
            g1_geom_model,
            g1_visual_model,
            g1_q0,
            g1_v0,
            parent_joint=ff_model.getFrameId("g1_freeflyer"),
        )
        g1_q0[:3] = np.array([-0.4, -0.4, 2.2])
        model, geom_model, visual_model, q0, v0 = mergeModels(
            model,
            geom_model,
            visual_model,
            q0,
            v0,
            g1_model,
            g1_geom_model,
            g1_visual_model,
            g1_q0,
            g1_v0,
        )

    if args.add_robots or args.add_talos:
        # --> talos
        ff_model, ff_geom_model, ff_visual_model, ff_q0 = createFreeflyerModel(
            "talos_freeflyer"
        )
        talos_robot = plr("talos_description")
        talos_model, talos_geom_model, talos_visual_model = (
            talos_robot.model,
            talos_robot.collision_model,
            talos_robot.visual_model,
        )
        talos_q0 = pin.neutral(talos_model)
        talos_v0 = np.zeros(talos_model.nv)
        addRobotName(talos_model, talos_geom_model, talos_visual_model, "talos")
        talos_model, talos_geom_model, talos_visual_model, talos_q0, talos_v0 = (
            mergeModels(
                ff_model,
                ff_geom_model,
                ff_visual_model,
                ff_q0,
                np.zeros(ff_model.nv),
                talos_model,
                talos_geom_model,
                talos_visual_model,
                talos_q0,
                talos_v0,
                parent_joint=ff_model.getFrameId("talos_freeflyer"),
            )
        )
        talos_q0[:3] = np.array([0.3, 0.3, 2.2])

        model, geom_model, visual_model, q0, v0 = mergeModels(
            talos_model,
            talos_geom_model,
            talos_visual_model,
            talos_q0,
            talos_v0,
            model,
            geom_model,
            visual_model,
            q0,
            v0,
        )

    if args.add_robots or args.add_go2:
        # --> go2
        ff_model, ff_geom_model, ff_visual_model, ff_q0 = createFreeflyerModel(
            "go2_freeflyer"
        )
        go2_robot = plr("go2_description")
        go2_model, go2_geom_model, go2_visual_model = (
            go2_robot.model,
            go2_robot.collision_model,
            go2_robot.visual_model,
        )
        go2_q0 = pin.neutral(go2_model)
        go2_v0 = np.zeros(go2_model.nv)
        addRobotName(go2_model, go2_geom_model, go2_visual_model, "go2")
        go2_model, go2_geom_model, go2_visual_model, go2_q0, go2_v0 = mergeModels(
            ff_model,
            ff_geom_model,
            ff_visual_model,
            ff_q0,
            np.zeros(ff_model.nv),
            go2_model,
            go2_geom_model,
            go2_visual_model,
            go2_q0,
            go2_v0,
            parent_joint=ff_model.getFrameId("go2_freeflyer"),
        )
        go2_q0[:3] = np.array([-0.3, 0.3, 2.2])

        model, geom_model, visual_model, q0, v0 = mergeModels(
            go2_model,
            go2_geom_model,
            go2_visual_model,
            go2_q0,
            go2_v0,
            model,
            geom_model,
            visual_model,
            q0,
            v0,
        )

    if args.add_robots or args.add_anymal:
        # --> anymal_d
        ff_model, ff_geom_model, ff_visual_model, ff_q0 = createFreeflyerModel(
            "anymal_d_freeflyer"
        )
        anymal_d_robot = plr("anymal_d_description")
        anymal_d_model, anymal_d_geom_model, anymal_d_visual_model = (
            anymal_d_robot.model,
            anymal_d_robot.collision_model,
            anymal_d_robot.visual_model,
        )
        anymal_d_q0 = pin.neutral(anymal_d_model)
        anymal_d_v0 = np.zeros(anymal_d_model.nv)
        addRobotName(
            anymal_d_model, anymal_d_geom_model, anymal_d_visual_model, "anymal_d"
        )
        (
            anymal_d_model,
            anymal_d_geom_model,
            anymal_d_visual_model,
            anymal_d_q0,
            anymal_d_v0,
        ) = mergeModels(
            ff_model,
            ff_geom_model,
            ff_visual_model,
            ff_q0,
            np.zeros(ff_model.nv),
            anymal_d_model,
            anymal_d_geom_model,
            anymal_d_visual_model,
            anymal_d_q0,
            anymal_d_v0,
            parent_joint=ff_model.getFrameId("anymal_d_freeflyer"),
        )
        anymal_d_q0[:3] = np.array([0.3, -0.3, 2.2])

        model, geom_model, visual_model, q0, v0 = mergeModels(
            anymal_d_model,
            anymal_d_geom_model,
            anymal_d_visual_model,
            anymal_d_q0,
            anymal_d_v0,
            model,
            geom_model,
            visual_model,
            q0,
            v0,
        )

    if args.random_init_vel:
        v0 = np.random.randn(model.nv)

    runSimpleSimulationFromModel(
        model, geom_model, visual_model, q0, v0, args, add_floor=True
    )

# Run simulation based on selected backend
elif args.backend == "mujoco":
    # MuJoCo backend doesn't support adding robots
    if (
        args.add_robots
        or args.add_g1
        or args.add_talos
        or args.add_go2
        or args.add_anymal
    ):
        print(
            "Warning: MuJoCo backend only supports pyramid simulation (no robots). Ignoring robot flags."
        )
        args.add_robots = False
        args.add_g1 = False
        args.add_talos = False
        args.add_go2 = False
        args.add_anymal = False

    def generate_pyramid_mjcf(mass, levels, cube_size=0.25, spacing=1e-2, friction=1.0):
        """
        Generate a MuJoCo XML file for a pyramid of cubes.

        Parameters:
        - mass: float, mass of each cube
        - levels: int, number of levels in the pyramid
        - cube_size: float, edge length of each cube (default 0.25)
        - spacing: float, gap between cubes (default 1e-2)
        - friction: float, friction coefficient for contact (default 1.0)

        Returns:
        - xml_string: str, MuJoCo XML model as a string
        """
        half_size = cube_size / 2.0

        xml_lines = []
        xml_lines.append('<mujoco model="pyramid">')
        xml_lines.append('  <option timestep="0.001" gravity="0 0 -9.81"/>')
        xml_lines.append('  <compiler angle="radian"/>')
        xml_lines.append("")
        xml_lines.append("  <default>")
        xml_lines.append(
            '    <geom type="box" size="{} {} {}" mass="{}" friction="{} 0.005 0.0001"/>'.format(
                half_size, half_size, half_size, mass, friction
            )
        )
        xml_lines.append("  </default>")
        xml_lines.append("")
        xml_lines.append("  <worldbody>")
        xml_lines.append("    <!-- Ground plane -->")
        xml_lines.append(
            '    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1" friction="{} 0.005 0.0001"/>'.format(
                friction
            )
        )
        xml_lines.append("")

        # Generate pyramid cubes
        for level in range(levels):
            n = levels - level
            z = (level + 0.5) * (cube_size + spacing)

            xml_lines.append("    <!-- Level {} -->".format(level))
            for i in range(n):
                for j in range(n):
                    # Center grid around (0,0)
                    x = (i - (n - 1) / 2.0) * (cube_size + spacing)
                    y = (j - (n - 1) / 2.0) * (cube_size + spacing)
                    name = "cube_{}_{}_{}".format(level, i, j)

                    # Add freejoint body with box geometry
                    xml_lines.append(
                        '    <body name="{}" pos="{} {} {}">'.format(name, x, y, z)
                    )
                    xml_lines.append("      <freejoint/>")
                    xml_lines.append(
                        '      <geom name="{}_geom" rgba="0.8 0.3 0.3 1"/>'.format(name)
                    )
                    xml_lines.append("    </body>")

        xml_lines.append("  </worldbody>")
        xml_lines.append("</mujoco>")

        return "\n".join(xml_lines)

    # Generate MuJoCo XML for pyramid
    # Get friction coefficient based on material
    friction_map = {
        "metal": 1.0,
        "wood": 0.6,
        "concrete": 0.8,
    }
    friction = friction_map.get(args.material, 1.0)

    xml_content = generate_pyramid_mjcf(
        mass=1.0, levels=4, cube_size=0.25, spacing=1e-2, friction=friction
    )

    import os
    import tempfile

    # Save to temporary file and run MuJoCo simulation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        runMujocoXML(xml_path, args)
    finally:
        # Clean up temporary file
        if os.path.exists(xml_path):
            os.remove(xml_path)
