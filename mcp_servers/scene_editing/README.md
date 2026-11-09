# 标注驱动场景编辑接口

core/adversarial 提供独立编辑、PDM 挑战与渲染实现，adapter/ 提供 AdAgent 接口，runtime/ 保存每次任务产物。
真实后端命令已配置；接口接入与真实场景验收分别记录，详见[攻防闭环后端接入说明](../../docs/EDITING_INTERFACES.md)。

- run_attack_editing：攻击车编辑、候选接受与冻结场景。
- run_ego_editing：内部冲突理解、候选构造/优化、选择、点编辑/修复与专家轨迹提交。

输入输出、后端命令协议及上下游示例见 [编辑接口文档](../../docs/EDITING_INTERFACES.md)。
命令配置为 adapter/settings.json 的 commands.attack_editing / commands.ego_editing。

测试：从项目根目录运行 python -m pytest mcp_servers/scene_editing/adapter/tests -q。
测试不代表真实算法、渲染、仿真或训练已经验证。

## 本地环境软链接

编辑、渲染及 PDM 的解释器统一从项目根目录的 `environments/` 引用：

- `environments/scene_editing`：攻击车与自车编辑环境。
- `environments/scene_render`：高斯渲染环境。
- `environments/simengine`：PDM 仿真环境。

这些目录可软链接到已安装的独立环境，例如在项目根目录执行：

```bash
ln -s /absolute/path/to/editing-environment environments/scene_editing
ln -s /absolute/path/to/render-environment environments/scene_render
ln -s /absolute/path/to/simengine-environment environments/simengine
```

目标环境需提前安装依赖；软链接不会复制环境。迁移项目后，将
`adapter/settings.json` 的 `commands` 解释器和
`core/adversarial/backend_settings.json` 的 `pdm.python`、`render.python`
更新为新项目下的绝对路径。`environments/` 已由根目录 `.gitignore` 排除。
