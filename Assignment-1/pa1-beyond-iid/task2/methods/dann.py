#! Domain-Adversarial Neural Network: binary domain head + gradient reversal
import torch
from torch import nn
import torch.nn.functional as F


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, alpha):
        #! Keep forward representations identical; only reverse their gradients.
        ctx.alpha = float(alpha)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def reverse_gradient(features, alpha):
    return ReverseGradient.apply(features, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, features):
        return self.network(features)


def dann_objective(source_features, source_logits, source_labels,
                   target_features, target_logits, step, steps,
                   discriminator, alpha, domain_weight=1.0):
    #! Class loss sees only labeled sources; target class logits are never used.
    cls_loss = F.cross_entropy(source_logits, source_labels)
    combined = torch.cat((source_features, target_features), dim=0)
    domain_logits = discriminator(reverse_gradient(combined, alpha))
    domain_labels = torch.cat((
        torch.zeros(len(source_features), dtype=torch.long, device=combined.device),
        torch.ones(len(target_features), dtype=torch.long, device=combined.device),
    ))
    domain_loss = F.cross_entropy(domain_logits, domain_labels)
    domain_accuracy = (domain_logits.detach().argmax(1) == domain_labels).float().mean()
    return cls_loss + domain_weight * domain_loss, {
        'classification': cls_loss, 'alignment': domain_loss,
        'domain_accuracy': domain_accuracy,
    }
