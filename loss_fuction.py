import torch as t
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, y_pred, y_true):
        intersection = (y_pred * y_true).sum()
        union = (y_pred + y_true).sum() - intersection
        iou = (intersection + 1e-5) / (t.abs(union) + 1e-5)
        iou_loss = 1 - iou
        return iou_loss

if __name__ == "__main__":
    y_truth = t.tensor([[0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 0]], dtype=t.float32)
    y_pred = t.tensor([[0.1, 0.2, 0.1],
                       [0.14, 0.1, 0.4],
                       [0.33, 0.5, 0.8]], dtype=t.float32)
    obj = IoULoss()
    loss_value = obj(y_truth, y_pred)
    print("IoULoss:", loss_value.item())
