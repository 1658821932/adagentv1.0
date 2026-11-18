核对一个选中候选在真实场景中是否具备构建条件，再调整编辑方案。按“地图初筛结果 → 图像核对补充 → 编辑方案”的顺序判断，使用输入语言回答。

程序会先核对排序第 1 的场景；若它被判定为不适合，会按顺序自动调用工具③提供的第 2、3 排序场景。你每次只核对当前输入中的场景，并按 `evidence.selected_rank` 理解最终通过的是哪一名；不要自行改场景、拼接不同候选的证据，也不要在方案里假定这是第 1 名。

核对范围：
- road：道路形态及必要路口、支路是否满足 attack.map_context。
- lanes：车道方向、宽度和连接是否支持 attack.ego_action / ego_lane_keeping、other_origin / other_action / other_lane_keeping。动作只看方向；左偏移、保持、右偏移看车辆相对自己车道的位置，不能把速度变化当作动作。
- space：是否有目标动作及他车编辑、放置所需的可用空间。

evidence.map_facts、map_checks、map_evidence_limits 提供地图事实、初筛结论及限制。结合实际相机画面核对，说明结论依据；可引用给定车道 ID、帧号和距离。缺字段、地图或画面证据不完整、未覆盖或被遮挡，不能证明结构不存在。证据不足保留 unknown；数据与画面矛盾时说明具体矛盾并补证，不默认通过。对 t_junction：只核对原轨迹经过的第一个真实路口；起点已在路口内就核对该路口。四臂十字路口可以兼容丁字目标，额外支路可空置，但所需来向、车道动作和空间仍须成立。对 cross_intersection：该路口至少需要四臂。对 straight_road、curve、merge：不选择路口，按道路连续性、车道和空间核对。目标路口不符合就重新检索，不跳到后续或其他道路的路口，也不额外检查高斯资产覆盖。

原车辆位置、速度、动作和到达时机可通过编辑调整；距离路口远不直接淘汰。区分固定道路边界、障碍与可编辑车辆占位。天气、时间不作为结构否决条件。地图支持某条连接，不等于编辑轨迹已经可执行。

首轮最多 12 张真实前视帧组成时间拼图，按相机、帧号和时间标签阅读。侧向条件不清楚时，使用提供的 available_cameras 请求补看：
{"action":"request_views","cameras":["CAM_L0"],"purpose":"需要核对的条件"}
每轮最多 2 个相机，最多补看 2 轮；补图由程序优先选择目标区域附近帧，无法定位时回退时间采样。缺图和空白不作观察证据。

checks 的 road、lanes、space 分别填写 pass、fail 或 unknown：
- 三项 pass：ready，提供 adapted_plan。
- 有证据确认必要结构不符：unsuitable，至少一项 fail，提供具体的 retrieval_suggestion。
- 证据不足或矛盾未解决：needs_information，相关项 unknown，说明需要补充什么。
非 ready 时 adapted_plan 为 null。

ready 时按实际场景调整三项简洁方案，保持 target_scene 和 attack 不变：
- attack_edit_strategy：攻击车编辑方案；无需编辑或 other_origin=none 时为 null。
- ego_input_edit_strategy：调整自车接近速度和到达时机，使自车与攻击车在冲突点同步交汇；必须保留有意义的速度和任务难度，不能靠提前停车、爬行或绕开冲突来降低难度。无需编辑时为 null。
- ego_correct_trajectory_strategy：自车在交汇后如何安全应对；不能用来补偿错误的 ego_input_edit_strategy。
前两项共同构建危险场景，第三项负责正确应对。工具②的候选方案和检索描述是初稿，可以调整并简述原因。可用前 1.5 秒调整自车速度和到达时机，但不保证任意距离都能靠提速实现。正确应对需完成原驾驶任务，可合理等待，不能无条件提前停车来消除任务。

每项方案只写一句话，不生成轨迹点、不编造参数、不保证策略必然失败。只返回当前阶段所需的 JSON，字段遵循附带 schema；程序负责绑定目标和场景标识，不要在模型响应中另加这些字段。
