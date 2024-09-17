# evaluation.py

from sklearn.metrics import classification_report, f1_score

def calculate_metrics(true_labels, pred_labels, class_labels, file=None):
    """실제 레이블과 예측 레이블을 비교하여 성능 지표를 계산합니다."""
    report = classification_report(true_labels, pred_labels, labels=class_labels, digits=4)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', labels=class_labels)

    if file:
        file.write(report)
        file.write(f'\nMacro-F1 Score: {macro_f1}\n')
    else:
        print(report)
        print('Macro-F1 Score:', macro_f1)
