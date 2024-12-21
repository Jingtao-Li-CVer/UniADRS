import numpy as np
import torch
import torch.nn.functional as F

def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.

    Args:
        precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
        num_anchors: int, number of equally spaced anchor points.

    Returns:
        precision_values: A `Tensor` of [num_anchors] equally spaced values
            in the interval precision_range.

        delta: The spacing between the values in precision_values.

    Raises:
        ValueError: If precision_range is invalid.
    """
    if len(precision_range) != 2:
        raise ValueError(
            "length of precision_range (%d) must be 2" % len(precision_range)
        )
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = np.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return torch.FloatTensor(precision_values), delta


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)


def weighted_ce_loss(labels, logits, thresholds, positive_weights=1.0, negative_weights=1.0, device='cuda:0'):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """

    
    labels = labels.squeeze(1)
    weight = torch.ones(labels.shape,).to(device)
    weight[labels == 0] *=  0.5
    weight[labels == 2] *=  0

    loss_t = 0.0
    for i in range((thresholds.shape[1])):
        threshs = (thresholds[:,i])
        weight2 = weight.clone()
        locs1 = torch.where((labels == 1) & (logits[:,0] <= threshs))
        locs2 = torch.where((labels == 0) & (logits[:,0] >= threshs))
        locs3 = torch.where((labels == 0))
        weight2[locs1[0]] = weight2[locs2[0]] = -1*torch.log(threshs+1e-8)
        weight2[locs3[0]] = weight2[locs3[0]]*(negative_weights[0,i])*0.2

        loss_t1 = F.binary_cross_entropy(logits[:,0], labels.float(), reduction='none')
        loss_t1 = loss_t1 * weight2

        loss_t = (loss_t + loss_t1.mean())/2.0

    return loss_t


def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.

    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors