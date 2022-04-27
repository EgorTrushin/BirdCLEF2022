import numpy as np
from sklearn import metrics
from .data import SCORED_MASK


def comp_metric(y_true, y_pred, epsilon=1e-9):
    """ Function to calculate competition metric in an sklearn like fashion

    Args:
        y_true{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Ground truth (correct) target values.
        y_pred{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Estimated targets as returned by a classifier.
    Returns:
        The single calculated score representative of this competitions evaluation
    """

    # Get representative confusion matrices for each label
    mlbl_cms = metrics.multilabel_confusion_matrix(y_true, y_pred)

    # Get two scores (TP and TN SCORES)
    tp_scores = np.array([
        mlbl_cm[1, 1]/(epsilon+mlbl_cm[:, 1].sum()) \
        for mlbl_cm in mlbl_cms
        ])
    tn_scores = np.array([
        mlbl_cm[0, 0]/(epsilon+mlbl_cm[:, 0].sum()) \
        for mlbl_cm in mlbl_cms
        ])

    # Get average
    tp_mean = tp_scores.mean()
    tn_mean = tn_scores.mean()

    return round((tp_mean+tn_mean)/2, 8)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred["clipwise_output"].cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.m02 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.2, epsilon=1e-9)
        self.m03 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.3, epsilon=1e-9)
        self.m035 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.35, epsilon=1e-9)
        self.m04 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.4, epsilon=1e-9)
        self.m045 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.45, epsilon=1e-9)
        self.m05 = comp_metric(np.array(self.y_true)[:,SCORED_MASK], np.array(self.y_pred)[:,SCORED_MASK] > 0.5, epsilon=1e-9)

        return {
            "m_0.2" : self.m02,
            "m_0.3" : self.m03,
            "m_0.35" : self.m035,
            "m_0.4" : self.m04,
            "m_0.45" : self.m045,
            "m_0.5" : self.m05,
        }
