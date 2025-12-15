# utils/metrics.py
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def calculate_metrics(y_true, y_score):
    y_pred = (y_score > 0.5).astype(int)
    ap = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]) if (y_true == 0).any() else 1.0
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]) if (y_true == 1).any() else 1.0
    auc = roc_auc_score(y_true, y_score)
    return ap, acc, auc, r_acc, f_acc