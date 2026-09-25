#! Task 4 post-hoc novelty scores: higher is more unknown; no unknown data used here
import numpy as np

SCORE_NAMES = ('MSP', 'MLS', 'Energy', 'Mahalanobis')


def fit_diagonal_mahalanobis(features, labels, num_classes=10, epsilon=1e-6):
    #! One class mean per label; one shared WITHIN-CLASS pooled diagonal covariance.
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError('Features must be [N,D] and labels [N].')
    if (num_classes < 2 or len(features) <= num_classes or not np.isfinite(features).all()
            or np.any(labels < 0) or np.any(labels >= num_classes)):
        raise ValueError('Invalid training features, class indices or population size.')

    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts < 2):
        raise ValueError('At least two unaugmented training examples are required per class.')
    means = np.stack([features[labels == label].mean(axis=0)
                      for label in range(num_classes)])
    residuals = features - means[labels]
    #! Pooled unbiased within-class covariance, diagonal only, regularized by 1e-6.
    variance = np.sum(residuals ** 2, axis=0) / (len(features) - num_classes)
    variance = variance + float(epsilon)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError('Non-finite / nonpositive covariance diagonal.')
    return means, variance, counts


def novelty_scores(logits, features, means, variance, block_size=512):
    #! Exactly the same frozen example outputs are reused for ALL scores.
    logits = np.asarray(logits, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if (logits.ndim != 2 or features.ndim != 2 or len(logits) != len(features)
            or means.shape != (logits.shape[1], features.shape[1])
            or variance.shape != (features.shape[1],)
            or not all(np.isfinite(a).all() for a in (logits, features, means, variance))
            or np.any(variance <= 0) or block_size <= 0):
        raise ValueError('Output, Mahalanobis reference or precision mismatch.')

    max_z = logits.max(axis=1)
    logsumexp_z = np.logaddexp.reduce(logits, axis=1)
    scores = {
        'MSP': 1.0 - np.exp(max_z - logsumexp_z),
        'MLS': -max_z,
        'Energy': -logsumexp_z,
    }
    inv_variance = 1.0 / variance
    mah = np.empty(len(features), dtype=np.float64)
    for begin in range(0, len(features), block_size):
        batch = features[begin:begin + block_size]
        differences = batch[:, None, :] - means[None, :, :]
        distances = np.einsum('ncd,d,ncd->nc', differences, inv_variance,
                              differences, optimize=True)
        mah[begin:begin + len(batch)] = distances.min(axis=1)
    scores['Mahalanobis'] = mah
    if any(not np.isfinite(score).all() for score in scores.values()):
        raise FloatingPointError('A novelty score contains NaN or infinity.')
    return scores


def known_validation_threshold(scores, percentile=0.95):
    #! Only known CIFAR-10 validation scores determine the fixed operating point.
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Threshold calibration requires finite known validation scores.')
    if not 0 < percentile < 1:
        raise ValueError('Percentile must lie strictly between zero and one.')
    return float(np.quantile(values, percentile, method='linear'))
