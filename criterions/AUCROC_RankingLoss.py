import torch
from torch import nn
from .functional_tools import *

class AUCROC_RankingLoss(nn.Module,):
    """
    1.0 The loss is implemented as the pixel-level optimization accroding to the paper
    "A Unified Remote Sensing Anomaly Detector Across Modalities and Scenes via Deviation Relationship Learning" 
    https://arxiv.org/abs/2310.07511
    
    2.0 Many basic codes are borrowed from the 'https://pytext.readthedocs.io/en/master/_modules/pytext/loss/loss.html#AUCPRHingeLoss'

    3.0 weighted_ce_loss function in functional_tools.py is the main difference compared to the traditional hinge loss or sigmoid loss,
        where better large-margin performance is achieved without any hyperparameter-tuning.
    """

    def __init__(self, num_classes, num_anchors, fpr_range_lower, fpr_range_upper, device):
        """Args:
        config: Config containing `fpr_range_lower`, `fpr_range_upper`,
            `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.fpr_range = (
            fpr_range_lower,
            fpr_range_upper,
        )
        self.device = device

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # fpr_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.fpr_values, self.delta = range_to_anchors_and_delta(
            self.fpr_range, self.num_anchors
        )
        self.fpr_values = self.fpr_values.to(device)

        self.negative_weight = 0.5
        self.biases = nn.Parameter(
                torch.FloatTensor(1, 6).zero_()
            )
        
        self.lambdas = nn.Parameter(
                torch.FloatTensor(1, 6).data.fill_(1))
  

    def forward(self, logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        labels = targets
        weights = torch.ones(labels.shape).to(self.device)
        lambdas = lagrange_multiplier(self.lambdas)

        thresholds = torch.tensor([0.9,0.85,0.8,0.75,0.7,0.6]).reshape((1,6)).to(self.device)

        ce_loss = weighted_ce_loss(
            labels.unsqueeze(-1),
            (logits.unsqueeze(-1)), 
            thresholds = thresholds,
            positive_weights = 1.0,
            negative_weights = lambdas,
            device = self.device,
        )

        class_priors = build_class_priors(labels.unsqueeze(1), weights=weights.unsqueeze(1))

        # lambda_term: Tensor[C, K]
        lambda_term = class_priors.unsqueeze(-1) * (
            (self.fpr_values)
        )

        per_anchor_loss =  ce_loss - lambda_term.mean()
        per_anchor_loss = per_anchor_loss.clamp(min=0)

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]

        loss = per_anchor_loss * self.delta * 100
        return loss

 
    def _prepare_labels_weights(self, logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """

        N, C = logits.size()
        t = torch.FloatTensor(N, C).zero_().to(self.device).to(torch.int64)
        labels = t.scatter(1, targets.unsqueeze(1).data.to(torch.int64), 1)

        if weights is None:
            weights = torch.FloatTensor(N).data.fill_(1.0).to(self.device)

        if weights.dim() == 1:
            weights.unsqueeze_(-1).to(self.device)

        return labels, weights.to(self.device)