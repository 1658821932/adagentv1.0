import torch


def instance_pair_rigidity_loss(source_points, dense_motion, instance_labels, pairs_per_instance=256):
    if source_points.shape != (*instance_labels.shape, 3):
        raise ValueError('instance labels must match source points as [B,S,H,W]')
    if dense_motion.shape[:2] != source_points.shape[:2] or dense_motion.shape[3:] != source_points.shape[2:]:
        raise ValueError('dense motion must have shape [B,S,K,H,W,3]')
    instance_losses = []
    for batch_index in range(source_points.shape[0]):
        for source_index in range(source_points.shape[1]):
            labels = instance_labels[batch_index, source_index].reshape(-1)
            points = source_points[batch_index, source_index].detach().reshape(-1, 3).float()
            motion = dense_motion[batch_index, source_index].flatten(1, 2).float()
            for instance_id in labels.unique().tolist():
                if instance_id <= 0:
                    continue
                members = torch.where(labels == instance_id)[0]
                if members.numel() < 2:
                    continue
                first = torch.randint(members.numel(), (pairs_per_instance,), device=members.device)
                second = (first + torch.randint(1, members.numel(), (pairs_per_instance,), device=members.device)) % members.numel()
                first, second = members[first], members[second]
                original_offset = points[first] - points[second]
                original_distance = original_offset.norm(dim=-1)
                moved_distance = (original_offset[None] + motion[:, first] - motion[:, second]).norm(dim=-1)
                instance_losses.append(
                    (moved_distance - original_distance[None]).abs().mean()
                    / original_distance.mean().clamp_min(1e-6)
                )
    if not instance_losses:
        return dense_motion.sum() * 0
    return torch.stack(instance_losses).mean()
