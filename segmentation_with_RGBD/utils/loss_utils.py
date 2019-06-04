import torch.nn.functional as F
from utils.utils import get_class_weights


def cross_entropy_2d():
    def wrap(seg_preds, seg_targets, class_inputs=None, class_targets=None,
             lambda_1=1.0, lambda_2=None, weight=None, pixel_average=True):

        # If the dataset is SUN RGB-D use class normalization weights in order to introduce balance to calculated loss as the number
        # of classes in SUN RGB-D dataset are not uniformly distributed.
        n, c, h, w = seg_preds.size()
        if c == 37:
            weight = get_class_weights('sun')

        # Calculate segmentation loss
        seg_inputs = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_inputs = seg_inputs[seg_targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        # Exclude the 0-valued pixels from the loss calculation as 0 values represent the pixels that are not annotated.
        seg_targets_mask = seg_targets > 0
        # Subtract 1 from all classes, in the ground truth tensor, in order to match the network predictions.
        # Remember, in network predictions, label 0 corresponds to label 1 in ground truth.
        seg_targets = seg_targets[seg_targets_mask] - 1

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.cross_entropy(seg_inputs, seg_targets, weight=weight, size_average=False)

        # Average the calculated loss value over each labeled pixel in the ground-truth tensor
        if pixel_average:
            seg_loss /= seg_targets_mask.float().data.sum()
        loss = lambda_1 * seg_loss

        # If scene classification function is utilized, calculate class loss, multiply with coefficient, lambda_2, sum with total loss
        if lambda_2 is not None:
            # Calculate classification loss
            class_targets -= 1
            class_loss = F.cross_entropy(class_inputs, class_targets)
            # Combine losses
            loss += lambda_2 * class_loss

            seg_loss = seg_loss.item()
            class_loss = class_loss.item()
            return loss, seg_loss, class_loss
        return loss
    return wrap
