"""Absolute annotation-driven rigid assets, independent of reconstructed log drift."""
import math
import numpy as np


def appearance_timestamp(model, timestamp):
    """Clamp appearance/attitude sampling, never extrapolate a source log pose."""
    if model.static_in_log:
        return int(timestamp)
    times = model.log_timestamps
    lower = int(times.min().item()) + int(model.log_start_time)
    upper = int(times.max().item()) + int(model.log_start_time)
    return max(lower, min(upper, int(timestamp)))


def grounded_absolute_pose(xy, yaw, transform, road_map, height):
    if not np.isfinite(np.r_[xy, yaw, height]).all() or height <= 0:
        raise ValueError('Absolute target pose and box height must be finite and valid')
    center = (transform @ np.r_[xy, 0., 1.])[:3]
    ground = road_map.height_with_local_plane(*center[:2])
    center[2] = ground['height'] + height / 2
    heading = math.atan2(transform[1, 0], transform[0, 0]) + float(yaw)
    return center, heading, ground


class AbsoluteTargetOverride:
    """Place the canonical Gaussian at edited coordinates, retaining source roll/pitch.

    Source log mismatches remain diagnostics. Edited pose is read back independently;
    the original 0.5m replay limit is never widened or used to excuse placement drift.
    """
    def __init__(self, model, translation, yaw, timestamp):
        self.model = model
        self.translation = np.asarray(translation, float)
        self.yaw = float(yaw)
        self.timestamp = int(timestamp)
        self.appearance_time = appearance_timestamp(model, timestamp)
        self.readback = None

    def get_global_gaussians(self, timestamp):
        import torch
        from worldengine.render.mtgs.utils.gaussian_utils import quat_mult
        if int(timestamp) != self.timestamp:
            raise ValueError('Absolute target override belongs to a different frame')
        if self.model.static_in_log:
            q = self.model.log_quats
        else:
            q, _, _ = self.model._get_log_pose_from_timestamp(self.appearance_time)
        q = q / q.norm()
        w, x, y, z = q
        old_yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)).item()
        half = (self.yaw-old_yaw)/2
        rotation = q.new_tensor([math.cos(half),0.,0.,math.sin(half)])
        new_q = quat_mult(rotation[None],q[None])[0]
        trans = q.new_tensor(self.translation)
        # Call primitive transforms: upstream static models ignore explicit poses.
        result = dict(means=self.model.get_means(global_quat=new_q,global_trans=trans),
                      quats=self.model.get_quats(global_quat=new_q,global_trans=trans),
                      scales=self.model.get_scales(),opacities=self.model.get_opacity())
        a,b,c,d = new_q / new_q.norm()
        recovered_yaw = float(torch.atan2(2*(a*d+b*c),1-2*(c*c+d*d)).item())
        yaw_error = abs(math.atan2(math.sin(recovered_yaw-self.yaw),math.cos(recovered_yaw-self.yaw)))
        position_error = float(np.linalg.norm(trans.detach().cpu().numpy()-self.translation))
        self.readback = dict(position_error_m=position_error,yaw_error_rad=yaw_error,
                             appearance_timestamp=self.appearance_time,
                             appearance_clamped=self.appearance_time != self.timestamp)
        if position_error > 2e-4 or yaw_error > 2e-5:
            raise ValueError('Absolute edited target pose failed transform readback')
        return result

    def get_gaussian_rgbs(self, **kwargs):
        kwargs['timestamp'] = self.appearance_time
        return self.model.get_gaussian_rgbs(**kwargs)


def covered_models(models, timestamp):
    """Omit unedited actors outside source lifetime rather than extrapolate poses."""
    visible, absent = {}, []
    for name, model in models.items():
        dynamic = getattr(model,'exist_log',False) and not getattr(model,'static_in_log',True)
        if dynamic and appearance_timestamp(model,timestamp) != int(timestamp):
            absent.append(name)
        else:
            visible[name] = model
    return visible, absent
