#! DAN source-target multi-kernel Maximum Mean Discrepancy
import torch
import torch.nn.functional as F

KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0)


def multi_kernel_mmd2(source_features, target_features,
                      multipliers=KERNEL_MULTIPLIERS, eps=1e-8):
    #! Biased empirical MMD² = mean(Kss) + mean(Ktt) - 2 mean(Kst).
    #! Biased estimator matches the squared distance between empirical RKHS means.
    #! The 3 RBF kernels are SUMMED (not averaged) as specified by the assignment.
    
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError('MMD expects two feature matrices [batch, feature_dim]')
    
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError('Source and target feature dimensions differ')
    
    n_source, n_target = len(source_features), len(target_features)
    
    if n_source < 2 or n_target < 2:
        raise ValueError('MMD requires at least two examples per domain')

    #! float32 avoids low-precision distance errors when ResNet uses CUDA autocast.
    with torch.autocast(device_type=source_features.device.type, enabled=False):
        combined = torch.cat((source_features.float(), target_features.float()), dim=0)
        dist2 = torch.cdist(combined, combined, p=2).square()
        non_diagonal = ~torch.eye(len(combined), dtype=torch.bool, device=combined.device)
    
        #! Bandwidth is computed from THIS combined source-target batch, without gradients.
        median_dist2 = dist2.detach()[non_diagonal].median().clamp_min(eps)
        kernel = sum(torch.exp(-dist2 / (float(scale) * median_dist2))
                     for scale in multipliers)
    
        ss = kernel[:n_source, :n_source].mean()
        tt = kernel[n_source:, n_source:].mean()
        st = kernel[:n_source, n_source:].mean()
        mmd2 = ss + tt - 2.0 * st
    
        #! The population expression is nonnegative; clamp only rounding negatives.
        return mmd2.clamp_min(0.0)


def dan_objective(source_features, source_logits, source_labels,
                  target_features, target_logits, step, steps, lambda_mmd=1.0):
    #! Target logits are intentionally ignored: there are no target class labels.
    
    cls = F.cross_entropy(source_logits, source_labels)
    mmd2 = multi_kernel_mmd2(source_features, target_features)
    return cls + lambda_mmd * mmd2, {'classification': cls, 'alignment': mmd2}
