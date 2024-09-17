# evaluation.py

from sklearn.metrics import classification_report, f1_score

def calculate_metrics(true_labels, pred_labels):
    """Evaluate..."""
    labels = ['supports', 'refutes', 'not enough information']
    report = classification_report(true_labels, pred_labels, labels=labels, digits=4)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', labels=labels)
    print(report)
    print('Macro-F1 Score:', macro_f1)
