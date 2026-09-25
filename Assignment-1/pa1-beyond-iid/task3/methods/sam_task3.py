
#! Standard, non-adaptive SAM: two source-CE passes, one optimizer update
import torch
import torch.nn.functional as F


def sam_step(model, optimizer, images, labels, rho=0.05):
    #! The SAME prepared minibatch is used in both passes; BN modes are set by caller.

    if rho <= 0:
        raise ValueError('SAM radius must be positive')

    optimizer.zero_grad(set_to_none=True)
    clean_logits = model(images)
    clean_loss = F.cross_entropy(clean_logits, labels)

    if not bool(torch.isfinite(clean_loss).item()):
        raise FloatingPointError('Non-finite SAM first-pass source CE')

    predictions = clean_logits.detach().argmax(dim=1)
    clean_loss.backward()

    #! Global L2 norm over every trainable parameter; inf is a finite-check, NOT clipping.
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float('inf'), error_if_nonfinite=True
    )

    if gradient_norm.item() <= 0:
        raise FloatingPointError('Zero SAM gradient: normalized ascent is undefined')

    scale = float(rho) / gradient_norm
    offsets = []
    try:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    epsilon = parameter.grad.detach().clone().mul_(scale)
                    parameter.add_(epsilon)
                    offsets.append((parameter, epsilon))

        #! Clear first-pass gradients; differentiate ONLY the perturbed loss.
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = F.cross_entropy(model(images), labels)

        if not bool(torch.isfinite(perturbed_loss).item()):
            raise FloatingPointError('Non-finite SAM second-pass source CE')

        perturbed_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf'), error_if_nonfinite=True
        )
    finally:
        #! Restore unperturbed parameters EVEN IF the second pass fails.
        with torch.no_grad():
            for parameter, epsilon in offsets:
                parameter.sub_(epsilon)

    #! AdamW applies exactly ONCE at original theta, with gradient at theta + epsilon.
    optimizer.step()
    return {
        'clean_ce': clean_loss.detach().item(),
        'perturbed_ce': perturbed_loss.detach().item(),
        'first_gradient_norm': gradient_norm.detach().item(),
        'perturbation_l2': torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(epsilon) for _, epsilon in offsets
        ])).detach().item(),
        'predictions': predictions,
    }
