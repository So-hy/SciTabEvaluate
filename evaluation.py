# evaluation.py

from sklearn.metrics import classification_report, f1_score

def calculate_metrics(true_labels, pred_labels, labels, output_file=None, experiment_name=''):
    """실제 레이블과 예측 레이블을 비교하여 성능 지표를 계산하고, 결과를 파일로 저장합니다."""
    report = classification_report(true_labels, pred_labels, labels=labels, digits=4)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', labels=labels)
    
    result_text = f'=== {experiment_name} ===\n' + report + '\nMacro-F1 Score: {:.4f}\n\n'.format(macro_f1)
    print(result_text)
    
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(result_text + '\n')
