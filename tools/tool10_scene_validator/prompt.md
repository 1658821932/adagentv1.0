# 任务
根据提供的关键帧，检查本次编辑是否满足最初场景需求。逐帧观察，证据注明从 1 开始的帧序号。

# 检查维度
- requirement_consistency：道路情境、目标参与者和事件是否符合需求。
- geometry_valid：目标车位置、尺度、接地和道路关系是否合理。
- visual_valid：目标是否可辨认、跨帧连续，是否出现编辑引入的破碎或伪影。
- physics_valid：可见运动是否明显不合理；这是视觉判断，不是动力学验证。
- traffic_logic_valid：可见交互和行驶方向是否符合需求。

# 判断依据
目标车明显悬空、出道路或破碎时，将相应维度设为 false，并在 issues 写明帧和问题。
涉及切入时，检查车辆完成切入后的车道位置及末帧连续性；接近画面中心不能证明在车道内。
关键目标被遮挡到无法确认编辑效果时，判为不满足并说明原因。
只判断图像支持的现象；精确距离/尺度、自车制动和闭环策略表现须由轨迹或仿真核实。

# 已知底场景缺陷
{ignored_artifacts}
这些已知高斯渲染缺陷不单独计入编辑失败，可记入 ignored_reconstruction_artifacts；遮住关键目标的情况仍按上面的规则处理。

# 输出
只返回符合下列 Schema 的 JSON。evidence 写逐帧依据；issues 写需要修改的问题；全部维度通过时 issues 为空。
{response_schema}

# 最初场景需求
{requirement}
