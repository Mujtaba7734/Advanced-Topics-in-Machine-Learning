#! DAN-DG: source-only pairwise alignment, using the unchanged Task 2 MMD
import torch
import torch.nn.functional as F

from dan import multi_kernel_mmd2

SOURCE_PAIRS = ((0, 1), (0, 2), (1, 2))


def forward_features_logits(model, images):
    #! ResNet-18 pooled 512-D representation immediately before its classifier.
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = torch.flatten(model.avgpool(x), 1)
    return features, model.fc(features)


def dan_dg_objective(features, logits, labels, lambda_dg=1.0):
    #! Images ordered in groups of 8: Photo, Art Painting, Cartoon.

    if features.shape != (24, 512) or logits.shape != (24, 7) or labels.shape != (24,):
        raise ValueError('Expected [24,512] features, [24,7] logits and 24 source labels')

    groups = (features[:8], features[8:16], features[16:24])
    ce = F.cross_entropy(logits, labels)
    penalties = [multi_kernel_mmd2(groups[i], groups[j]) for i, j in SOURCE_PAIRS]
    pairwise_mmd = torch.stack(penalties).mean()
    return ce + float(lambda_dg) * pairwise_mmd, ce, pairwise_mmd, penalties
