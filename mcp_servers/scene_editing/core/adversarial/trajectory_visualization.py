"""Continuous time-colored trajectory views; no prescribed event timestamps."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon

from trajectory_checks import continuous_headings


def _trajectory(ax, points, times, norm, label, dashed=False, arrows=True, ends=True, lw=2.2):
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(segments, cmap='viridis', norm=norm,
        linewidths=lw, linestyles='dashed' if dashed else 'solid', label=label)
    collection.set_array((times[:-1]+times[1:])/2)
    ax.add_collection(collection)
    ax.update_datalim(points)
    ax.autoscale_view()
    if arrows:
        # Direction marks are spaced by traveled distance, not fixed times.
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if distance[-1] > 1e-6:
            for value in np.linspace(0, distance[-1], 7)[1:-1]:
                i = min(int(np.searchsorted(distance, value)), len(points)-2)
                if np.linalg.norm(points[i+1]-points[i]) > 1e-8:
                    ax.annotate('', xy=points[i+1], xytext=points[max(0, i-1)],
                        arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(norm(times[i])), lw=1.5))
    if ends:
        for i, name in [(0, 'start'), (-1, 'end')]:
            ax.plot(*points[i], 'o', color=plt.cm.viridis(norm(times[i])), ms=4)
            ax.annotate(label+' '+name, points[i], fontsize=7, xytext=(4, 4), textcoords='offset points')
    return collection


def _time_ticks(ax, points, times, norm, count=7):
    for i in np.linspace(0, len(times)-1, min(count, len(times))).round().astype(int):
        ax.plot(*points[i], 'o', ms=3.5, color=plt.cm.viridis(norm(times[i])),
            markeredgecolor='black', markeredgewidth=.5, zorder=4)
        ax.annotate(f'{times[i]:.0f}s', points[i], fontsize=6.5, xytext=(3, -8),
            textcoords='offset points', zorder=4)


def _vehicle_box(ax, x, y, length_m, width_m, angle, face, alpha=.5, lw=1.):
    vertices = np.array([[-length_m/2, -width_m/2], [length_m/2, -width_m/2],
                         [length_m/2, width_m/2], [-length_m/2, width_m/2]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = vertices@rotation.T+[x, y]
    ax.add_patch(Polygon(points, closed=True, facecolor=face, edgecolor='black',
        alpha=alpha, lw=lw, zorder=5))
    return points


def _emptiest_corner(ax, clouds, fraction):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w, h = (x1-x0)*fraction, (y1-y0)*fraction
    best, best_count = ('left', 'lower'), None
    for horizontal, cx in [('left', x0), ('right', x1-w)]:
        for vertical, cy in [('lower', y0), ('upper', y1-h)]:
            count = sum(int(((p[:, 0] >= cx) & (p[:, 0] <= cx+w) &
                             (p[:, 1] >= cy) & (p[:, 1] <= cy+h)).sum()) for p in clouds)
            if best_count is None or count < best_count:
                best, best_count = (horizontal, vertical), count
    return best


def _zoom_inset(ax, points_a, points_b, times, norm, closest, focus_a, focus_b,
                heading, relative_angles, target_heading, target_size, ego_size, ego_panel,
                relevant_actors=None, sync_title=None):
    clouds = [p for p in (points_a, points_b) if p is not None and len(p)]
    if ego_panel:
        clouds.append(np.array([[0., 0.]]))
    fraction = .46
    horizontal, vertical = _emptiest_corner(ax, clouds, fraction)
    margin = .025
    inset = ax.inset_axes([
        margin if horizontal == 'left' else 1-fraction-margin,
        margin if vertical == 'lower' else 1-fraction-margin, fraction, fraction])
    center = (focus_a+focus_b)/2
    half = max(8., np.linalg.norm(focus_a-focus_b)/2+1.6*max(list(ego_size)+list(target_size or (0., 0.))))
    _trajectory(inset, points_a, times, norm, None, arrows=False, ends=False, lw=2.6)
    if points_b is not None and len(points_b):
        _trajectory(inset, points_b, times, norm, None, dashed=True, arrows=False, ends=False, lw=2.2)
    inside = np.flatnonzero((np.abs(points_a[:, 0]-center[0]) < half) &
                            (np.abs(points_a[:, 1]-center[1]) < half))
    if len(inside):
        glyphs = set(np.array(inside)[np.linspace(0, len(inside)-1, min(len(inside), 4)).round().astype(int)])
    else:
        glyphs = set()
    glyphs.add(closest)
    glyph_list = sorted(glyphs)
    if glyph_list:
        label_offsets = {glyph_list[0]: ((4, -12), 'left'), glyph_list[-1]: ((-4, -12), 'right'),
            closest: ((4, 6), 'left')}
    else:
        label_offsets = {}
    boxes = [np.array([[center[0]-half, center[1]-half], [center[0]+half, center[1]+half]])]
    for i in glyph_list:
        solid = i == closest
        if target_size is not None:
            boxes.append(_vehicle_box(inset, *points_a[i], *target_size,
                relative_angles[i] if ego_panel else target_heading[i], 'orange',
                alpha=.55 if solid else .22, lw=1.4 if solid else .8))
        if not ego_panel and points_b is not None and len(points_b):
            boxes.append(_vehicle_box(inset, *points_b[i], *ego_size, heading[i], 'green',
                alpha=.45 if solid else .18, lw=1.2 if solid else .7))
        # Draw synchronized other actors at this EXACT same sample index (skip if missing)
        if not ego_panel and relevant_actors:
            for actor in relevant_actors:
                sample = next((s for s in actor.get('samples', []) if s['sample_index'] == i), None)
                if sample is not None:
                    boxes.append(_vehicle_box(inset, sample['x'], sample['y'], sample['length_m'], sample['width_m'],
                        sample['heading_rad'], 'skyblue', alpha=.45 if solid else .18, lw=1.0 if solid else .6))
                    if solid:
                        inset.annotate(f"Trk {actor['track_id']}", (sample['x'], sample['y']), fontsize=6,
                            color='darkslategray', xytext=(2, 2), textcoords='offset points', zorder=7)
        if i in label_offsets:
            offset, horizontal = label_offsets[i]
            inset.annotate(f'{times[i]:.1f}s', points_a[i], fontsize=6.5,
                xytext=offset, textcoords='offset points', zorder=6,
                horizontalalignment=horizontal, fontweight='bold' if solid else 'normal')
    if ego_panel:
        boxes.append(_vehicle_box(inset, 0, 0, *ego_size, 0, 'green', alpha=.5))
        inset.axhspan(-1, 1, color='gray', alpha=.12)
    extents = np.vstack(boxes)
    lo, hi = extents.min(axis=0)-1.6, extents.max(axis=0)+1.6
    fitted_center, fitted_half = (lo+hi)/2, float(max(hi-lo)/2)
    inset.set(xlim=(fitted_center[0]-fitted_half, fitted_center[0]+fitted_half),
        ylim=(fitted_center[1]-fitted_half, fitted_center[1]+fitted_half))
    inset.set_aspect('equal')
    inset.grid(alpha=.25)
    if sync_title:
        inset.set_title(sync_title, fontsize=7.5)
    else:
        inset.set_title('Closest approach, true vehicle size' if target_size is not None
            else 'Closest approach (zoom)', fontsize=7.5)
    inset.tick_params(labelsize=6.5)
    ax.indicate_inset_zoom(inset, edgecolor='gray', alpha=.6)


def _oriented_box_polygon(x, y, length_m, width_m, angle):
    vertices = np.array([[-length_m/2, -width_m/2], [length_m/2, -width_m/2],
                         [length_m/2, width_m/2], [-length_m/2, width_m/2]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = vertices@rotation.T+[x, y]
    return ShapelyPolygon(points)


def plot_pool_overview(path, candidates, times, baseline, ego, heading,
                       other_actor_context=None, target_size=None, ego_size=(4.5, 2.)):
    """Overview of up to 3 candidate pool trajectories side-by-side with identical scale,
    focused on the interaction area and screening other vehicles in the candidate window.
    """
    times, baseline, ego, heading = map(np.asarray, (times, baseline, ego, heading))
    n = max(1, min(3, len(candidates)))
    fig, axes = plt.subplots(1, n, figsize=(5.8*n, 6.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    norm = plt.Normalize(times[0], times[-1])

    # Interaction window across candidates: anchor strictly to first ego contact ± ~2.0s
    contact_times = [float(c['first_overlap_time_s']) for c in candidates[:n] if c.get('first_overlap_time_s') is not None]
    if not contact_times:
        contact_times = [float(c.get('contact_time_s', 15.0)) for c in candidates[:n]]
    t_min = max(0.0, min(contact_times) - 2.0)
    t_max = min(float(times[-1]), max(contact_times) + 2.0)
    window_idx = np.where((times >= t_min) & (times <= t_max))[0]
    if not len(window_idx):
        window_idx = np.arange(len(times))

    # Select relevant other actors during this interaction window (not global baseline)
    relevant_actors = []
    if other_actor_context and other_actor_context.get('actors'):
        for actor in other_actor_context['actors']:
            in_window_samples = [s for s in actor.get('samples', []) if s['sample_index'] in window_idx]
            if not in_window_samples:
                continue
            dists = []
            for s in in_window_samples:
                si = s['sample_index']
                for cand in candidates[:n]:
                    dists.append(np.linalg.norm(np.array([s['x'], s['y']]) - cand['xy'][si]))
                dists.append(np.linalg.norm(np.array([s['x'], s['y']]) - ego[si]))
            relevant_actors.append((float(min(dists)), actor))
        relevant_actors.sort(key=lambda x: x[0])
        relevant_actors = [a[1] for a in relevant_actors[:3]]

    # Focus bounds on interaction area across all candidates for unified scale
    # ONLY use points within the interaction window (± ~2.0s of contact) so distant other-car
    # trajectories do not pull the window down to y=-80m!
    x_pts = [ego[window_idx, 0]]
    y_pts = [ego[window_idx, 1]]
    for cand in candidates[:n]:
        pts = np.asarray(cand['xy'])
        x_pts.append(pts[window_idx, 0])
        y_pts.append(pts[window_idx, 1])
    for actor in relevant_actors:
        s_pts = np.array([[s['x'], s['y']] for s in actor.get('samples', []) if s['sample_index'] in window_idx])
        if len(s_pts):
            x_pts.append(s_pts[:, 0])
            y_pts.append(s_pts[:, 1])

    min_x, max_x = float(np.min(np.concatenate(x_pts))), float(np.max(np.concatenate(x_pts)))
    min_y, max_y = float(np.min(np.concatenate(y_pts))), float(np.max(np.concatenate(y_pts)))
    span_x, span_y = max_x - min_x, max_y - min_y
    max_span = max(span_x, span_y, 14.0)
    mid_x, mid_y = (min_x + max_x)/2, (min_y + max_y)/2
    pad = max_span * 0.55
    xlim = (mid_x - pad, mid_x + pad)
    ylim = (mid_y - pad, mid_y + pad)

    last_collection = None
    for idx, (ax, cand) in enumerate(zip(axes, candidates[:n])):
        target = np.asarray(cand['xy'])
        for actor in relevant_actors:
            samples = actor['samples']
            indices = np.array([s['sample_index'] for s in samples])
            splits = np.flatnonzero(np.diff(indices) > 1) + 1
            for grp in np.split(samples, splits):
                pts = np.array([[s['x'], s['y']] for s in grp])
                if len(pts) > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color='darkslategray', alpha=0.45, lw=1.0, ls=':')

            # Synchronized vehicle box: must match EXACT sample_index, skipped when missing!
            t_contact = cand.get('first_overlap_time_s')
            if t_contact is not None:
                ci = int(round(float(t_contact)*10))
                exact_s = next((s for s in samples if s['sample_index'] == ci), None)
                if exact_s is not None:
                    _vehicle_box(ax, exact_s['x'], exact_s['y'], exact_s['length_m'], exact_s['width_m'],
                                 exact_s['heading_rad'], 'lightgray', alpha=0.4, lw=0.8)
                    ax.annotate(f"Trk {actor['track_id']}", (exact_s['x'], exact_s['y']),
                                fontsize=6, color='dimgray', xytext=(2, 2), textcoords='offset points')

        _trajectory(ax, ego, times, norm, 'Ego', dashed=True, lw=1.8, arrows=False, ends=False)
        last_collection = _trajectory(ax, target, times, norm, 'Target', lw=2.2, arrows=True, ends=True)
        _time_ticks(ax, target, times, norm, count=5)

        t_contact = cand.get('first_overlap_time_s')
        if t_contact is not None:
            ci = min(len(times)-1, max(0, int(round(float(t_contact)*10))))
            if target_size:
                th = cand.get('target_heading')
                if th is None:
                    th = continuous_headings(target, float(heading[0]))
                _vehicle_box(ax, target[ci, 0], target[ci, 1], target_size[0], target_size[1], th[ci], 'orange', alpha=0.6, lw=1.2)
            _vehicle_box(ax, ego[ci, 0], ego[ci, 1], ego_size[0], ego_size[1], heading[ci], 'green', alpha=0.5, lw=1.2)
            ax.annotate(f"first contact {times[ci]:.1f}s", (target[ci, 0], target[ci, 1]),
                        fontsize=7, fontweight='bold', color='crimson', xytext=(5, 5), textcoords='offset points')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel('Forward x (m)', fontsize=8)
        if idx == 0:
            ax.set_ylabel('Left y (m)', fontsize=8)

        seed_id = cand.get('seed_id', f'seed_{idx+1:03d}')
        t0 = cand.get('t0', 0.0)
        dt = cand.get('dt', 0.0)
        tail_from = cand.get('tail_from', 0.0)
        c_time = f"{cand['first_overlap_time_s']:.1f}s" if cand.get('first_overlap_time_s') is not None else "None"
        num_screen = cand.get('result', {}).get('numerical_screen', {}).get('metrics', {})
        max_acc = num_screen.get('max_acceleration_mps2')
        acc_text = f'{max_acc:.1f} m/s²' if max_acc is not None else 'unavailable'
        other_screen = cand.get('other_actor_screen') or cand.get('result', {}).get('other_actor_screen') or {}
        other_stat = other_screen.get('status', 'passed')
        coverage = other_screen.get('coverage', 'partial')
        feasibility = cand.get('active_feasibility', cand.get('status', 'unknown'))
        active_screen = 'relative' if 'relative_screen' in cand.get('result', {}) else 'absolute'

        card_text = (f"{seed_id}\n"
                     f"feasibility: {feasibility} ({active_screen} limits)\n"
                     f"goal: contact @ {c_time}\n"
                     f"onset t0={t0:.1f}s, dt={dt:.1f}s, tail={tail_from:.1f}s\n"
                     f"max absolute accel: {acc_text}\n"
                     f"other vehicles: {other_stat} (coverage={coverage})")
        ax.text(0.03, 0.97, card_text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
        ax.set_title(f"Candidate {seed_id}", fontsize=10, fontweight='bold')

    if last_collection:
        fig.colorbar(last_collection, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08,
                     label='Time (s); arrows indicate travel direction; dashed=ego, solid=candidate')
    fig.suptitle('Constructor Candidate Pool Overview (identical scale across all candidates)\n'
                 'Screened against all observed other vehicles; center distance != body clearance',
                 fontsize=11)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def plot_trajectory_view(path, times, target, ego, heading, title,
                         target_heading=None, target_size=None, ego_size=(4.5, 2.),
                         other_actor_context=None, relative_baseline=None,
                         event_time_s=None):
    from trajectory_checks import _series, relative_allowance, LIMITS

    times, target, ego, heading = map(np.asarray, (times, target, ego, heading))
    if target_heading is None:
        target_heading = continuous_headings(target, float(heading[0]))
    norm = plt.Normalize(times[0], times[-1])
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.5), constrained_layout=True)

    delta = target-ego
    relative = np.column_stack((np.cos(heading)*delta[:, 0]+np.sin(heading)*delta[:, 1],
        -np.sin(heading)*delta[:, 0]+np.cos(heading)*delta[:, 1]))
    relative_angles = target_heading-heading

    if event_time_s is not None:
        closest = int(np.argmin(np.abs(times - float(event_time_s))))
    else:
        closest = int(np.argmin(np.linalg.norm(relative, axis=1)))

    clearances = None
    if target_size is not None and ego_size is not None:
        clearances = np.zeros(len(times), dtype=float)
        for i in range(len(times)):
            p_tgt = _oriented_box_polygon(target[i, 0], target[i, 1], target_size[0], target_size[1], target_heading[i])
            p_ego = _oriented_box_polygon(ego[i, 0], ego[i, 1], ego_size[0], ego_size[1], heading[i])
            clearances[i] = p_tgt.distance(p_ego)

    relevant_actors = []
    if other_actor_context and other_actor_context.get('actors'):
        for actor in other_actor_context['actors']:
            samples = actor.get('samples', [])
            if not samples:
                continue
            pts = np.array([[s['x'], s['y']] for s in samples])
            dists = np.linalg.norm(pts - target[[s['sample_index'] for s in samples]], axis=1)
            relevant_actors.append((float(np.min(dists)), actor))
        relevant_actors.sort(key=lambda x: x[0])
        relevant_actors = [a[1] for a in relevant_actors[:3]]

    collection = _trajectory(axes[0, 0], target, times, norm, 'Target')
    _trajectory(axes[0, 0], ego, times, norm, 'Ego', dashed=True)

    for actor in relevant_actors:
        samples = actor['samples']
        indices = np.array([s['sample_index'] for s in samples])
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        for grp in np.split(samples, splits):
            pts = np.array([[s['x'], s['y']] for s in grp])
            if len(pts) > 1:
                axes[0, 0].plot(pts[:, 0], pts[:, 1], color='darkslategray', alpha=0.45, lw=1.0, ls=':')
        sync_sample = next((s for s in samples if s['sample_index'] == closest), None)
        if sync_sample is not None:
            _vehicle_box(axes[0, 0], sync_sample['x'], sync_sample['y'], sync_sample['length_m'],
                         sync_sample['width_m'], sync_sample['heading_rad'], 'skyblue', alpha=0.55, lw=1.0)
            axes[0, 0].annotate(f"Trk {actor['track_id']} @ {times[closest]:.1f}s",
                                (sync_sample['x'], sync_sample['y']), fontsize=6.5, color='darkslategray',
                                xytext=(3, 3), textcoords='offset points', fontweight='bold')
        else:
            nearest_sample = min(samples, key=lambda s: abs(s['sample_index'] - closest))
            _vehicle_box(axes[0, 0], nearest_sample['x'], nearest_sample['y'], nearest_sample['length_m'],
                         nearest_sample['width_m'], nearest_sample['heading_rad'], 'lightsteelblue', alpha=0.25, lw=0.7)
            axes[0, 0].annotate(f"Trk {actor['track_id']} (no obs @ {times[closest]:.1f}s)",
                                (nearest_sample['x'], nearest_sample['y']), fontsize=6, color='dimgray',
                                xytext=(3, 3), textcoords='offset points')

    axes[0, 0].plot([target[closest, 0], ego[closest, 0]], [target[closest, 1], ego[closest, 1]],
        color='crimson', lw=1., ls=':', zorder=4)
    dist_label = f'center dist {np.linalg.norm(relative[closest]):.1f}m'
    if clearances is not None:
        dist_label += f' (body clr {clearances[closest]:.1f}m)'
    dist_label += f' @ {times[closest]:.1f}s'
    if event_time_s is not None:
        dist_label = f'event {times[closest]:.1f}s: ' + dist_label
    axes[0, 0].annotate(dist_label, (target[closest]+ego[closest])/2, fontsize=7, color='crimson', xytext=(4, 4),
        textcoords='offset points', zorder=4)
    _time_ticks(axes[0, 0], target, times, norm)
    axes[0, 0].set(title='Fixed frame: solid target / dashed ego / dotted other', xlabel='Forward x (m)', ylabel='Left y (m)')
    axes[0, 0].set_aspect('equal', adjustable='datalim')

    _trajectory(axes[0, 1], relative, times, norm, 'Relative target')
    axes[0, 1].plot(0, 0, 'ks', label='Ego origin')
    axes[0, 1].axhspan(-1, 1, color='gray', alpha=.12, label='Path proxy, not road boundaries')
    axes[0, 1].plot([0, relative[closest, 0]], [0, relative[closest, 1]], color='crimson', lw=1., ls=':', zorder=4)
    axes[0, 1].annotate(dist_label, relative[closest]/2, fontsize=7, color='crimson', xytext=(4, 4), textcoords='offset points', zorder=4)
    _time_ticks(axes[0, 1], relative, times, norm)
    axes[0, 1].set(title='Current ego frame: target motion over time', xlabel='Relative forward (m)', ylabel='Relative left (m)')
    axes[0, 1].set_aspect('equal', adjustable='datalim')

    axes[0, 0].autoscale_view(); axes[0, 1].autoscale_view()
    sync_title_0 = f'Synchronized @ {times[closest]:.1f}s (true size)' if target_size is not None else f'Synchronized @ {times[closest]:.1f}s'
    sync_title_1 = f'Relative sync @ {times[closest]:.1f}s'
    _zoom_inset(axes[0, 0], target, ego, times, norm, closest,
        target[closest], ego[closest], heading, relative_angles, target_heading, target_size, ego_size, ego_panel=False,
        relevant_actors=relevant_actors, sync_title=sync_title_0)
    _zoom_inset(axes[0, 1], relative, None, times, norm, closest,
        relative[closest], np.array([0., 0.]), heading, relative_angles, target_heading, target_size, ego_size, ego_panel=True,
        relevant_actors=None, sync_title=sync_title_1)

    axes[1, 0].plot(times, relative[:, 0], label='Forward center gap')
    axes[1, 0].plot(times, relative[:, 1], label='Left center offset')
    axes[1, 0].plot(times, np.linalg.norm(relative, axis=1), label='Center distance', color='crimson', ls=':', lw=1.2)
    if clearances is not None:
        axes[1, 0].plot(times, clearances, label='Vehicle body clearance', color='purple', ls='--', lw=1.2)
    axes[1, 0].set(xlabel='Time (s)', ylabel='Distance (m)', title='Synchronized relative geometry')

    axes[1, 1].plot(times, np.linalg.norm(np.gradient(target, times, axis=0), axis=1), label='Target speed', color='teal', lw=1.2)
    axes[1, 1].plot(times, np.linalg.norm(np.gradient(ego, times, axis=0), axis=1), label='Ego speed', color='darkgreen', ls='--', lw=1.1)
    axes[1, 1].axhline(LIMITS['max_speed_mps'], color='gray', ls=':', label='Max speed limit (40m/s)')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Speed (m/s)', title='Speed matching & speed limit')

    # Row 2: Constraint Evidence Curves (同源 series/mask and relative allowance)
    target_3d = np.column_stack([times, target])
    s = _series(target_3d)
    mid_t = s['mid_t']
    accel = np.linalg.norm(s['acceleration'], axis=1)

    if relative_baseline is not None:
        base_3d = np.column_stack([times, relative_baseline])
        allow = relative_allowance(base_3d, LIMITS)
        allow_accel = allow['max_acceleration_mps2']
        allow_lat = allow['max_lateral_acceleration_mps2']
        allow_yaw = allow['max_yaw_rate_radps']
        allow_title = 'per-time relative allowance'
    else:
        allow_accel = np.full_like(mid_t, LIMITS['max_acceleration_mps2'])
        allow_lat = np.full_like(mid_t, LIMITS['max_lateral_acceleration_mps2'])
        allow_yaw = np.full_like(mid_t, LIMITS['max_yaw_rate_radps'])
        allow_title = 'fixed absolute limits'

    axes[2, 0].plot(mid_t, accel, label='Target total accel', color='crimson', lw=1.3)
    axes[2, 0].plot(mid_t, allow_accel, label=f'Total accel allowance ({allow_title})', color='gray', ls='--', lw=1.2)
    axes[2, 0].set(xlabel='Time (s)', ylabel='Acceleration (m/s²)', title=f'Total acceleration vs {allow_title}')

    axes[2, 1].plot(mid_t, np.where(s['moving'], s['lateral'], np.nan), label='Target lateral accel (moving mask)', color='darkorange', lw=1.2)
    axes[2, 1].plot(mid_t, allow_lat, label='Lateral allowance', color='saddlebrown', ls='--', lw=1.1)
    axes[2, 1].plot(mid_t, np.where(s['turning'], s['yaw'], np.nan), label='Target yaw rate (turning mask)', color='purple', lw=1.2)
    axes[2, 1].plot(mid_t, allow_yaw, label='Yaw rate allowance', color='indigo', ls=':', lw=1.1)
    axes[2, 1].set(xlabel='Time (s)', ylabel='Lat accel (m/s²) / Yaw rate (rad/s)', title=f'Lateral & yaw constraints vs {allow_title}')

    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
        ax.legend(fontsize=7)
    fig.colorbar(collection, ax=list(axes[0]), label='Time (s); arrows show motion direction')
    if target_size is not None:
        note_lines = ('Full sampled trajectories; closest distance is center-to-center, distinct from vehicle body clearance',
            'Insets use recorded vehicle dimensions and tangent-estimated headings, not measured poses;',
            'faint boxes show the same vehicle at other times; other-actor observations broken at missing frames')
    else:
        note_lines = ('Full sampled trajectories; closest distance is center-to-center, distinct from vehicle body clearance',
            'Insets zoom the closest approach; no fixed action schedule or recovered road boundaries')
    fig.suptitle(title+'\n'+'\n'.join(note_lines), fontsize=10.5)
    fig.savefig(path, dpi=135)
    plt.close(fig)
