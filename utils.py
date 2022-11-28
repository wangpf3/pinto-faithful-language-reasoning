import logging
import torch
import torch.nn.functional as F

def get_logger(name, log_path=None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if log_path:
        handler = logging.FileHandler(log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self,):
        super(LabelSmoothingLoss, self).__init__()

    def linear_combination(self, x, y, smoothing):
        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target, smoothing, nll=None):

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1) / n
        if nll is None:
            nll = F.nll_loss(log_preds, target, reduction='none')
        return self.linear_combination(loss, nll, smoothing).mean()