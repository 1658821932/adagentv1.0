/**
 * adagent 步骤精简结果渲染模块
 */

const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

const SUITABILITY = { ready: '可编辑', needs_information: '信息不足', unsuitable: '不适合编辑' };
const OPERATION = { add_vehicle: '新增车辆', edit_existing: '修改现有车辆' };
const MANEUVER = { stationary: '静止阻挡', left_cut_in: '左侧切入' };

function editOptions(result) {
  return Array.isArray(result.edit_options)
    ? result.edit_options.filter(option => option && typeof option === 'object' && !Array.isArray(option))
    : [];
}

export function firstSentence(text, limit = 90) {
  if (typeof text !== 'string') return '';
  const cut = text.split(/[。；\n]/)[0].trim();
  return cut.length > limit ? cut.slice(0, limit) + '…' : cut;
}

function truncateText(text, limit = 260) {
  if (typeof text !== 'string') return '';
  const norm = text.replace(/\s+/g, ' ').trim();
  if (!norm) return '';
  return norm.length > limit ? norm.slice(0, limit) + '…' : norm;
}

export function formatMetricValue(val) {
  if (typeof val === 'number' && Number.isFinite(val)) {
    if (Number.isInteger(val)) return String(val);
    const abs = Math.abs(val);
    if (abs > 0 && abs < 0.001) return val.toExponential(2);
    return String(Number(val.toFixed(3)));
  }
  if (typeof val === 'boolean') return val ? '是' : '否';
  return String(val ?? '');
}

function formatLoss(val) {
  if (typeof val !== 'number' || !Number.isFinite(val)) return formatMetricValue(val);
  if (Number.isInteger(val)) return String(val);
  const abs = Math.abs(val);
  if (abs > 0 && abs < 0.0001) return val.toExponential(2);
  return String(Number(val.toFixed(4)));
}

function formatErrorText(err) {
  if (!err) return '';
  if (typeof err === 'string') return err;
  const parts = [err.type, err.message].filter(Boolean);
  return parts.join(': ') || JSON.stringify(err);
}

function attackPolicySummary(evaluation) {
  if (!evaluation || typeof evaluation !== 'object') return [];
  const verdict = evaluation.policy_attack_success === true
    ? 'VADv2 自测：攻击成功'
    : evaluation.status === 'passed'
      ? 'VADv2 自测：攻击未成功'
      : 'VADv2 自测：证据不足';
  const collision = evaluation.target_collision_failed === true
    ? '目标碰撞已发生'
    : evaluation.target_collision_failed === false ? '未确认目标碰撞' : '目标碰撞证据不足';
  const road = evaluation.drivable_area;
  if (road?.status === 'failed') {
    const time = Number.isFinite(road.first_violation_time_s)
      ? `（首次 t=${formatMetricValue(road.first_violation_time_s)}s）`
      : '';
    return [verdict, collision, `驶出可行驶区${time}`];
  }
  return [verdict, collision, road?.status === 'passed' ? '可行驶区检查通过' : '可行驶区证据不足'];
}

/** 每个工具产出的核心结论（不截断、无 [object Object]） */
export function summaryFor(s) {
  const r = s?.output?.result;
  if (!r || typeof r !== 'object') {
    const errText = formatErrorText(s?.output?.error);
    return errText ? truncateText(errText, 220) : '';
  }

  if (s.tool === 'failure_analyzer') {
    const fa = r.failure_analysis || {};
    const types = Array.isArray(fa.failure_type)
      ? fa.failure_type.filter(Boolean).join(' / ')
      : (fa.failure_type || '');
    const reason = truncateText(fa.failure_reason, 260);
    return [types ? `[${types}]` : '', reason].filter(Boolean).join(' ');
  }

  if (s.tool === 'scene_instance_generator') {
    const req = truncateText(r.scene_requirement, 180);
    const goal = truncateText(r.target_effect, 140);
    const count = Array.isArray(r.candidates) ? r.candidates.length : 0;
    return [req, goal ? `目标：${goal}` : '', count ? `${count} 个候选方案` : ''].filter(Boolean).join(' · ');
  }

  if (s.tool === 'video_retriever') {
    const req = truncateText(r.requirement || s?.input?.requirement, 120);
    const score = typeof r.match_score === 'number' ? `匹配度 ${r.match_score.toFixed(3)}` : '';
    const token = r.scene_ref?.scene_token || r.scene_token;
    const scene = token ? `场景 ${token}` : '';
    const seg = r.source_segment
      && typeof r.source_segment.start_time === 'number'
      && typeof r.source_segment.end_time === 'number'
      ? `片段 ${r.source_segment.start_time}s–${r.source_segment.end_time}s`
      : '';
    return [req, score, scene, seg].filter(Boolean).join(' · ');
  }

  if (s.tool === 'edit_planner') {
    const verdict = SUITABILITY[r.suitability] || r.suitability || '';
    const options = editOptions(r);
    if (options.length) {
      const requirement = truncateText(r.scene_requirement || s?.input?.scene_requirement, 180)
        || truncateText(Array.isArray(r.reasons) ? r.reasons.filter(Boolean).join('；') : '', 180);
      const goal = truncateText(r.target_effect || s?.input?.target_effect, 140);
      return [verdict, requirement, goal ? `目标：${goal}` : '', `${options.length} 个候选方案（不分排名）`]
        .filter(Boolean).join(' · ');
    }
    const plan = r.edit_plan || {};
    const op = OPERATION[plan.operation] || plan.operation || '';
    const man = MANEUVER[plan.maneuver] || plan.maneuver || '';
    const target = plan.target_track ? ` ${plan.target_track}` : '';
    const opLabel = [op ? op + target : '', man].filter(Boolean).join(' · ');
    const head = [verdict, opLabel].filter(Boolean).join(' · ');
    const body = truncateText(plan.instruction, 260)
      || truncateText(Array.isArray(r.reasons) ? r.reasons.filter(Boolean).join('；') : '', 260);
    return [head, body].filter(Boolean).join(' — ');
  }

  if (s.tool === 'attack_vehicle_editor') {
    if (r.status === 'completed') {
      const op = OPERATION[r.operation] || r.operation || '';
      const target = r.target_track ? `目标 ${r.target_track}` : '';
      const cand = r.selected_candidate_id ? `候选 ${r.selected_candidate_id}` : '';
      const sim = r.simulation_input ? '已导出仿真包' : '';
      return ['已完成攻击场景冻结', op, target, cand, sim, ...attackPolicySummary(r.policy_evaluation)]
        .filter(Boolean).join(' · ');
    }
    const status = r.status === 'unable_to_complete' ? '未能完成' : (r.status || '未完成');
    return [status, truncateText(r.reason || r.stop_reason, 220)].filter(Boolean).join(' — ');
  }

  if (s.tool === 'ego_trajectory_editor') {
    if (r.status === 'completed') {
      const cand = r.selected_candidate_id ? `候选 ${r.selected_candidate_id}` : '';
      const src = r.selected_source ? `来源 ${String(r.selected_source).toUpperCase()}` : '';
      const sim = r.simulation_input ? '已导出仿真包' : '';
      const render = r.render_status?.status === 'failed' ? '回放生成失败' : '';
      return ['已完成自车专家轨迹', cand, src, sim, render].filter(Boolean).join(' · ');
    }
    const status = r.status === 'defense_not_found'
      ? '未找到可行防御'
      : r.status === 'unable_to_complete'
        ? '未能完成'
        : (r.status || '未完成');
    return [status, truncateText(r.reason || r.stop_reason, 220)].filter(Boolean).join(' — ');
  }

  if (s.tool === 'policy_runner') {
    const c = r.collision_report || {};
    const bits = [];
    if (typeof c.target_collision_detected === 'boolean') {
      const timeNote = c.target_collision_detected && typeof c.first_target_collision_time_s === 'number'
        ? `（t=${formatMetricValue(c.first_target_collision_time_s)}s）`
        : '';
      bits.push((c.target_collision_detected ? '目标碰撞已发生' : '未发生目标碰撞') + timeNote);
    } else if (r.pdm_challenge) {
      bits.push('PDM 挑战完成');
    }
    if (typeof c.min_target_clearance_m === 'number') {
      bits.push(`最小间距 ${formatMetricValue(c.min_target_clearance_m)}m`);
    }
    if (r.evaluation_scope && typeof r.evaluation_scope === 'object') {
      const sc = Array.isArray(r.evaluation_scope.scenes) ? r.evaluation_scope.scenes[0] : null;
      if (sc && typeof sc.start_time_s === 'number' && typeof sc.end_time_s === 'number') {
        const frames = sc.max_frame_per_scene || sc.sampling_poses;
        bits.push(`评测窗口 ${sc.start_time_s}s–${sc.end_time_s}s` + (frames ? `（${frames}帧）` : ''));
      } else if (r.evaluation_scope.scope) {
        bits.push(String(r.evaluation_scope.scope));
      }
    } else if (typeof r.evaluation_scope === 'string' && r.evaluation_scope) {
      bits.push(truncateText(r.evaluation_scope, 60));
    }
    if (r.candidate_rewards_present) bits.push('已导出候选奖励');
    return bits.join(' · ');
  }

  if (s.tool === 'failure_case_filter') {
    const count = r.scene_count || 1;
    const hitCount = Array.isArray(r.failure_cases)
      ? r.failure_cases.length
      : (r.is_failure_case ? 1 : 0);
    const head = r.is_failure_case === true ? '命中零分失败案例' : '未命中失败案例';
    const rows = Array.isArray(r.scene_results) && r.scene_results.length
      ? r.scene_results
      : (Array.isArray(r.failure_cases) ? r.failure_cases : []);
    const scores = rows
      .map(row => row?.pdm_score)
      .filter(v => typeof v === 'number' && Number.isFinite(v));
    const scoreText = scores.length
      ? ` · PDM 总分 ${scores.slice(0, 3).map(formatMetricValue).join(' / ')}`
      : '';
    return `${head} · 共评测 ${count} 个场景（失败 ${hitCount} 个）${scoreText}`;
  }

  if (s.tool === 'policy_trainer') {
    const mode = r.training_mode ? String(r.training_mode).toUpperCase() : '微调';
    const audit = r.training_audit || {};
    const bits = [`${mode} 微调完成`];
    if (typeof audit.iterations === 'number') bits.push(`迭代 ${audit.iterations} 轮`);
    if (Array.isArray(audit.loss_trace) && audit.loss_trace.length >= 2) {
      const first = audit.loss_trace[0];
      const last = audit.loss_trace[audit.loss_trace.length - 1];
      if (typeof first === 'number' && typeof last === 'number') {
        bits.push(`Loss ${formatLoss(first)} → ${formatLoss(last)}`);
      }
    } else if (typeof audit.final_loss === 'number') {
      bits.push(`最终 Loss ${formatLoss(audit.final_loss)}`);
    }
    if (typeof audit.parameters_updated === 'boolean') {
      bits.push(audit.parameters_updated ? '参数已更新' : '参数未更新');
    }
    return bits.join(' · ');
  }

  for (const path of ['summary', 'message', 'requirement', 'scene_requirement']) {
    const v = path.split('.').reduce((o, k) => (o && typeof o === 'object' ? o[k] : undefined), r);
    if (typeof v === 'string' && v) return truncateText(v, 220);
  }
  return '';
}

/** 仅渲染单一、干净的结果简述框（去除冗余指标网格、几何安全检查、耗时分解与路径列表） */
export function renderStepStructured(s) {
  const errText = formatErrorText(s?.output?.error);
  if (errText) {
    return `<div class="structured-error"><div class="raw-title">执行报错</div><pre>${escape(errText)}</pre></div>`;
  }

  const text = summaryFor(s);
  const r = s?.output?.result;
  const extras = [];

  if (r && typeof r === 'object') {
    if (s.tool === 'failure_analyzer') {
      const factors = r.failure_analysis?.key_scene_factors;
      if (Array.isArray(factors) && factors.length) {
        extras.push(`<ul class="summary-list">${factors.map(item => `<li>${escape(item)}</li>`).join('')}</ul>`);
      }
    }

    if (s.tool === 'scene_instance_generator' && Array.isArray(r.candidates) && r.candidates.length) {
      const items = r.candidates.map((c, idx) => {
        const title = c.id ? `候选 ${c.id}` : `候选方案 ${String(idx + 1).padStart(2, '0')}`;
        const focus = c.focus ? `（${c.focus}）` : '';
        const desc = c.base_video_requirement || c.note || '';
        return `<li><b>${escape(title)}</b>${escape(focus)}${desc ? `：${escape(desc)}` : ''}</li>`;
      }).join('');
      extras.push(`<ul class="summary-list">${items}</ul>`);
    }

    if (s.tool === 'edit_planner') {
      const options = editOptions(r);
      if (options.length) {
        const items = options.map((option, idx) => {
          const title = `候选方案 ${String(idx + 1).padStart(2, '0')}`;
          const id = option.option_id ? `（${option.option_id}）` : '';
          const labels = [
            OPERATION[option.operation] || option.operation,
            MANEUVER[option.maneuver] || option.maneuver,
            option.target_track ? `目标车辆 ${option.target_track}` : '',
          ].filter(Boolean).join(' · ');
          const details = [
            option.rationale ? `理由：${option.rationale}` : '',
            option.uncertainty ? `待核实：${option.uncertainty}` : '',
          ].filter(Boolean).join('；');
          return `<li><b>${escape(title)}</b>${escape(id)}${labels ? ` · ${escape(labels)}` : ''}`
            + `${option.instruction ? `：${escape(option.instruction)}` : ''}`
            + `${details ? `<div class="summary-note">${escape(details)}</div>` : ''}</li>`;
        }).join('');
        extras.push(`<ul class="summary-list">${items}</ul>`);
      }
      if (r.suitability !== 'ready' && r.retrieval_suggestion) {
        extras.push(`<p class="summary-note">重检索建议：${escape(r.retrieval_suggestion)}</p>`);
      }
    }
  }

  const inner = [
    text ? `<p>${escape(text)}</p>` : '',
    ...extras,
  ].filter(Boolean).join('');

  if (!inner) return '';
  return `<div class="summary-block"><div class="raw-title">结果简述</div><div class="step-summary">${inner}</div></div>`;
}

export function renderStepMetrics() {
  return '';
}

export function renderStepDeliverables() {
  return '';
}

export function renderRunDeliverables() {
  return '';
}

export function bindDeliverableCopy() {}
