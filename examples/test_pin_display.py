import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.animation  # 引入 meshcat 的动画模块

# --- 1. 模型加载与设置 (保持不变) ---
pinocchio_model_dir = Path(__file__).parent.parent / "third_party" / "pinocchio" / "models"
model_path = pinocchio_model_dir / "example-robot-data/robots"
mesh_dir = pinocchio_model_dir
urdf_filename = "solo.urdf"
urdf_model_path = model_path / "solo_description/robots" / urdf_filename

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

try:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
except ImportError as err:
    print("Error while initializing the viewer.")
    sys.exit(0)

viz.loadViewerModel()

# --- 2. 场景设置 (保持不变) ---
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)

# 添加 Convex 几何体
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex

if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    geometry = pin.GeometryObject("convex", 0, placement, convex)
    geometry.meshColor = np.ones(4)
    geometry.overrideMaterial = True
    geometry.meshMaterial = pin.GeometryPhongMaterial()
    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
    geometry.meshMaterial.meshShininess = 0.8
    visual_model.addGeometryObject(geometry)
    viz.rebuildData() # 重建数据以包含新物体

# 第二个机器人 (保持不变，但注意它不会包含在下面的动画录制中，除非也加入逻辑)
viz2 = MeshcatVisualizer(model, collision_model, visual_model)
viz2.initViewer(viz.viewer)
viz2.loadViewerModel(rootNodeName="pinocchio2")
q = q0.copy()
q[1] = 1.0
viz2.display(q)

# --- 3. 仿真循环 (保持不变) ---
q1 = np.array(
    [0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
)
v0 = np.random.randn(model.nv) * 2
data = viz.data
pin.forwardKinematics(model, data, q1, v0)
frame_id = model.getFrameId("HR_FOOT")
viz.display()
viz.drawFrameVelocities(frame_id=frame_id)

model.gravity.linear[:] = 0.0
dt = 0.01

# 原始仿真循环 (稍微修改以仅返回数据，不在计算时实时display，提高速度)
def sim_loop():
    tau0 = np.zeros(model.nv)
    qs = [q1]
    vs = [v0]
    nsteps = 100
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        viz.drawFrameVelocities(frame_id=frame_id)
    return qs, vs

qs, vs = sim_loop()

# --- 4. 关键修改：生成 Meshcat 动画对象 (PyDrake 风格) ---

def generate_meshcat_animation(viz, qs, dt):
    """
    将轨迹 qs 转换为 Meshcat 的 Animation 对象并上传到浏览器。
    这样可以在网页端使用进度条（Slider）和播放控件。
    """
    print("Generating Meshcat animation...")
    
    # 1. 创建 Animation 对象
    # default_framerate 决定了网页端默认的播放速度
    fps = int(1.0 / dt)
    anim = meshcat.animation.Animation(default_framerate=fps)

    # 2. 遍历每一帧数据
    for frame_index, q in enumerate(qs):
        # 更新 Pinocchio 的运动学
        pin.forwardKinematics(viz.model, viz.data, q)
        # 更新所有几何体的放置位置 (Visual Objects)
        pin.updateGeometryPlacements(viz.model, viz.data, viz.visual_model, viz.visual_data)

        # 3. 记录该帧中每个 Visual Object 的位置
        # 'at_frame' 上下文管理器用于指定当前是哪一帧
        with anim.at_frame(viz.viewer, frame_index) as frame:
            for i, visual_obj in enumerate(viz.visual_model.geometryObjects):
                # 获取 Pinocchio 计算出的该物体的全局变换矩阵 (SE3 -> 4x4 Matrix)
                M = viz.visual_data.oMg[i]
                T = M.homogeneous
                
                # 获取该物体在 Meshcat 中的路径名称
                # Pinocchio 的 getViewerNodeName 通常返回 "pinocchio/base_link" 这样的路径
                node_name = viz.getViewerNodeName(visual_obj, pin.GeometryType.VISUAL)
                
                # 将变换矩阵设置到动画帧中对应的节点
                frame[node_name].set_transform(T)

    # 4. 将构建好的动画发送给 Meshcat 服务器
    viz.viewer.set_animation(anim)
    print(f"Animation uploaded! Open the Meshcat URL and look for 'Animations' controls.")

# 执行动画生成
generate_meshcat_animation(viz, qs, dt)

# 打印 URL 提醒用户打开
# 如果是在本地运行，通常是 http://127.0.0.1:7000/static/
# 如果是在 Jupyter 中，Meshcat 通常会自动内嵌
input("Press Enter after opening the Meshcat viewer to see the animation...")