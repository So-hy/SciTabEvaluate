# main.py

import torch
from data_utils import load_data, split_data, linearize_table
from model_utils import load_model, generate_prediction
from evaluation import calculate_metrics

def map_prediction_to_label(prediction):
    """Mapping Label"""
    if 'True' in prediction or 'A' in prediction:
        return 'supports'
    elif 'False' in prediction or 'B' in prediction:
        return 'refutes'
    elif 'Unknown' in prediction or 'C' in prediction:
        return 'not enough information'
    else:
        return 'unknown'

def zero_shot_evaluate(model, tokenizer, test_data, device):
    """Zero-shot Evaluate..."""
    true_labels = []
    pred_labels = []
    for sample in test_data:
        input_text = linearize_table(sample)
        prediction = generate_prediction(model, tokenizer, input_text, device)
        predicted_label = map_prediction_to_label(prediction)
        true_labels.append(sample['label'])
        pred_labels.append(predicted_label)
    return true_labels, pred_labels

def in_context_evaluate(model, tokenizer, test_data, example_data, label_mapping, device):
    """In-Context Learning Evaluate..."""
    # 예시 입력 구성
    example_text = ''
    for sample in example_data:
        input_text = linearize_table(sample)
        answer = label_mapping.get(sample['label'], 'Unknown')
        example_text += input_text + ' ' + answer + '\n\n---\n\n'

    true_labels = []
    pred_labels = []
    for sample in test_data:
        input_text = linearize_table(sample)
        full_input = example_text + input_text
        prediction = generate_prediction(model, tokenizer, full_input, device)
        predicted_label = map_prediction_to_label(prediction)
        true_labels.append(sample['label'])
        pred_labels.append(predicted_label)
    return true_labels, pred_labels

if __name__ == "__main__":
    # 재현성을 위해 랜덤 시드 설정
    torch.manual_seed(42)

    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 및 토크나이저 로드
    model_name = 'google/flan-t5-large'
    tokenizer, model = load_model(model_name, device)

    # 데이터 로드
    data_file = 'sci_tab.json'  # 데이터 파일 경로 지정
    data = load_data(data_file)

    # 데이터 분할
    example_data, test_data = split_data(data)

    print(f"예시 데이터 개수: {len(example_data)}")
    print(f"테스트 데이터 개수: {len(test_data)}")

    # 레이블 맵핑
    label_mapping = {
        'supports': 'True',
        'refutes': 'False',
        'not enough information': 'Unknown'
    }

    print("Zero-shot 평가를 시작합니다...")
    true_labels_zs, pred_labels_zs = zero_shot_evaluate(model, tokenizer, test_data, device)
    print("Zero-shot 평가 완료.\n")

    print("In-Context Learning 평가를 시작합니다...")
    true_labels_ic, pred_labels_ic = in_context_evaluate(model, tokenizer, test_data, example_data, label_mapping, device)
    print("In-Context Learning 평가 완료.\n")

    print("Zero-shot 평가 결과:")
    calculate_metrics(true_labels_zs, pred_labels_zs)

    print("\nIn-Context Learning 평가 결과:")
    calculate_metrics(true_labels_ic, pred_labels_ic)
