import pinocchio as pin
import coal
import numpy as np
import meshcat.animation


def sub_sample(xs, duration, fps):
    nb_frames = len(xs)
    nb_subframes = int(duration * fps)
    if nb_frames < nb_subframes:
        return xs
    else:
        step = nb_frames // nb_subframes
        xs_sub = [xs[i] for i in range(0, nb_frames, step)]
        return xs_sub


def add_floor(
    collision_model: pin.GeometryModel,
    visual_model: pin.GeometryModel | None,
    color: tuple[float, ...] = (0.5, 0.5, 0.5, 0.4),
    use_cube_for_visual=True,
    visual_cube_height=0.01,
) -> int | tuple[int, int]:
    """
    在 Pinocchio 的碰撞模型和（可选的）视觉模型中添加一个表示地板的几何体。
    如果模型中已经包含了名为 "floor" 的对象，则仅更新它的颜色。
    为了解决渲染器难以绘制无限平面的问题，视觉模型默认使用具有一定厚度的长方体，
    而碰撞模型则使用保证数值稳定性的半空间（无限大平面）。

    参数:
        collision_model: Pinocchio 碰撞相关的几何模型。
        visual_model: Pinocchio 视觉相关的几何模型 (可为 None)。
        color: 地板 RGBA 颜色，默认为半透明灰色 (0.5, 0.5, 0.5, 0.4)。
        use_cube_for_visual: 是否在视觉模型中使用 Box (长方体) 代替无限平面进行渲染。
        visual_cube_height: 设定的视觉 Box 厚度（默认 0.01 米）。

    返回:
        如果只更新了 collision_model，则返回其几何体 ID (int)；
        如果同时包含了 visual_model，则返回包含两者几何体 ID 的元组 (tuple[int, int])。
    """
    # lookup floor in existing collision model
    for gobj in collision_model.geometryObjects:
        gobj: pin.GeometryObject
        if gobj.name == "floor":
            gobj.meshColor[:] = color
            if visual_model is not None:
                visual_model.geometryObjects[
                    visual_model.getGeometryId("floor")
                ].meshColor[:] = color
            return collision_model.getGeometryId("floor")

    # if not, add color
    floor_collision_shape = coal.Halfspace(0, 0, 1, 0)
    M = pin.SE3.Identity()
    floor_offset = np.array([0.0, 0.0, -0.01])
    M.translation += floor_offset
    floor_collision_object = pin.GeometryObject("floor", 0, 0, M, floor_collision_shape)
    floor_collision_object.meshColor[:] = color
    coll_gid = collision_model.addGeometryObject(floor_collision_object)

    if visual_model is None:
        return coll_gid

    if use_cube_for_visual:
        floor_visual_shape = coal.Box(20, 20, visual_cube_height)
        Mvis = pin.SE3.Identity()
        Mvis.translation[:] = (0.0, 0.0, -visual_cube_height / 2)
        Mvis.translation += floor_offset
        floor_visual_object = pin.GeometryObject(
            "floor", 0, 0, Mvis, floor_visual_shape
        )
        floor_visual_object.meshColor[:] = color
    else:
        floor_visual_object = floor_collision_object.copy()
        floor_visual_object.meshColor[:] = color
    visu_gid = visual_model.addGeometryObject(floor_visual_object)
    return coll_gid, visu_gid


def generate_meshcat_animation(viz, qs, dt):
    """
    将轨迹 qs 转换为 Meshcat 的 Animation 对象并上传到浏览器。
    这样可以在网页端使用进度条（Slider）和播放控件。
    """
    # 1. 创建 Animation 对象
    # default_framerate 决定了网页端默认的播放速度
    fps = int(1.0 / dt)
    anim = meshcat.animation.Animation(default_framerate=fps)

    # 2. 遍历每一帧数据
    for frame_index, q in enumerate(qs):
        # 更新 Pinocchio 的运动学
        pin.forwardKinematics(viz.model, viz.data, q)
        # 更新所有几何体的放置位置 (Visual Objects)
        pin.updateGeometryPlacements(
            viz.model, viz.data, viz.visual_model, viz.visual_data
        )

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
