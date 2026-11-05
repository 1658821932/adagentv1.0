# 标注驱动场景编辑接口

core/ 预留独立编辑项目，adapter/ 提供 AdAgent 接口，runtime/ 保存每次任务产物。
当前不包含或执行真实编辑算法；两个工具已注册，real 在未配置后端时明确报错。

- run_attack_editing：攻击车编辑、候选接受与冻结场景。
- run_ego_editing：内部冲突理解、候选构造/优化、选择、点编辑/修复与专家轨迹提交。

输入输出、后端命令协议及上下游示例见 [编辑接口文档](../../docs/EDITING_INTERFACES.md)。
命令配置为 adapter/settings.json 的 commands.attack_editing / commands.ego_editing。

测试：从项目根目录运行 python -m pytest mcp_servers/scene_editing/adapter/tests -q。
测试不代表真实算法、渲染、仿真或训练已经验证。
