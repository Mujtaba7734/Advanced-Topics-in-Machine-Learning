#! Conditional adversarial network: full multilinear feature-class interaction
import torch
from torch import nn
import torch.nn.functional as F


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        #! Forward identity; reverse only the gradient into features AND probabilities.
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def reverse_gradient(x, alpha):
    return ReverseGradient.apply(x, alpha)


def conditional_representation(features, logits):
    #! Full outer product: batch x 512 x 7 -> batch x 3584. NO detach on either path.
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError('CDAN expects ResNet-18 pre-classifier features [N,512]')
    if logits.ndim != 2 or logits.shape != (features.shape[0], 7):
        raise ValueError('CDAN expects seven-class logits for every feature vector')
    probabilities = F.softmax(logits.float(), dim=1)
    return torch.bmm(features.float().unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, conditional_features):
        return self.network(conditional_features)


def cdan_objective(source_features, source_logits, source_labels,
                   target_features, target_logits, step, steps,
                   discriminator, alpha, domain_weight=1.0):
    #! Source CE only; target logits supply soft class predictions, NEVER labels.
    cls_loss = F.cross_entropy(source_logits, source_labels)
    source_conditional = conditional_representation(source_features, source_logits)
    target_conditional = conditional_representation(target_features, target_logits)
    combined = torch.cat((source_conditional, target_conditional), dim=0)
    domain_logits = discriminator(reverse_gradient(combined, alpha))
    domain_labels = torch.cat((
        torch.zeros(len(source_features), dtype=torch.long, device=combined.device),
        torch.ones(len(target_features), dtype=torch.long, device=combined.device),
    ))
    #! NO entropy weights, confidence filters, or extra conditioning variants.
    domain_loss = F.cross_entropy(domain_logits, domain_labels)
    domain_accuracy = (domain_logits.detach().argmax(1) == domain_labels).float().mean()
    return cls_loss + domain_weight * domain_loss, {
        'classification': cls_loss,
        'alignment': domain_loss,
        'domain_accuracy': domain_accuracy,
    }
