# main.py

import torch
from data_utils import load_data, split_data, filter_data_by_labels, linearize_table
from model_utils import load_model, generate_prediction
from evaluation import calculate_metrics
from tqdm import tqdm

def map_prediction_to_label(prediction, labels_to_include):
    # 모델의 출력을 소문자로 변환하고 앞뒤 공백을 제거
    prediction = prediction.lower().strip()

    # 모델 출력에 따라 레이블 매핑
    if 'supports' in prediction or prediction == 'a' or prediction == 'yes':
        label = 'supports'
    elif 'refutes' in prediction or prediction == 'b' or prediction == 'no':  # "No"를 refutes로 처리
        label = 'refutes'
    elif 'not enough information' in prediction or 'unknown' in prediction or 'cannot determine' in prediction or prediction == 'c' or 'not enough info' in prediction:
        label = 'not enough information'
    else:
        label = 'not enough information'  # supports/refutes가 아니면 NEI로 처리

    # 해당 레이블이 포함되지 않으면 기본적으로 'not enough information' 처리
    if label not in labels_to_include:
        label = 'not enough information'
    
    return label


def zero_shot_evaluate(model, tokenizer, test_data, labels_to_include, device):
    """Zero-shot 평가를 수행합니다."""
    true_labels = []
    pred_labels = []
    for sample in tqdm(test_data, desc="Evaluating", unit="sample"):
        input_text = linearize_table(sample)
        prediction = generate_prediction(model, tokenizer, input_text, device)
        predicted_label = map_prediction_to_label(prediction, labels_to_include)
        true_labels.append(sample['label'])
        pred_labels.append(predicted_label)
    return true_labels, pred_labels

def in_context_evaluate(model, tokenizer, test_data, example_data, label_mapping, labels_to_include, device):
    """In-Context Learning 평가를 수행합니다."""
    # 예시 입력 구성 (3개 예시 사용)
    example_text = ''
    for sample in example_data[:3]:
        input_text = linearize_table(sample)
        answer = label_mapping.get(sample['label'], 'Unknown')
        example_text += input_text + ' ' + answer + '\n\n---\n\n'

    true_labels = []
    pred_labels = []
    for sample in tqdm(test_data, desc="Evaluating", unit="sample"):
        input_text = linearize_table(sample)
        full_input = example_text + input_text
        prediction = generate_prediction(model, tokenizer, full_input, device)
        predicted_label = map_prediction_to_label(prediction, labels_to_include)
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

    
    # 데이터 분할 (예시로 사용할 3개 샘플)
    example_data, test_data = split_data(data, num_examples=3)

    print(f"예시 데이터 개수: {len(example_data)}")
    print(f"테스트 데이터 개수: {len(test_data)}")
    
    nei_count = sum(1 for sample in test_data if sample['label'] == 'not enough information')
    print(f"테스트 데이터에서 'not enough information' 레이블의 샘플 수: {nei_count}")
    
    # 레이블 맵핑
    label_mapping = {
        'supports': 'Supports',
        'refutes': 'Refutes',
        'not enough information': 'Not enough information'
    }

    # 결과를 저장할 파일 경로 지정
    output_file = 'evaluation_results.txt'

    # 기존 파일을 초기화하기 위해 열고 닫음
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('SCITAB 평가 결과\n\n')
        
    # **2분류 실험 (NEI 제외)**
    labels_2class = ['supports', 'refutes']
    test_data_2class = filter_data_by_labels(test_data, labels_2class)
    example_data_2class = filter_data_by_labels(example_data, labels_2class)

    print(f"\n2분류 실험: Zero-shot 평가를 시작합니다...")
    true_labels_zs_2, pred_labels_zs_2 = zero_shot_evaluate(model, tokenizer, test_data_2class, labels_2class, device)
    print("2분류 실험: Zero-shot 평가 완료.\n")

    print("2분류 실험: In-Context Learning 평가를 시작합니다...")
    true_labels_ic_2, pred_labels_ic_2 = in_context_evaluate(model, tokenizer, test_data_2class, example_data_2class, label_mapping, labels_2class, device)
    print("2분류 실험: In-Context Learning 평가 완료.\n")

    print("2분류 실험: Zero-shot 평가 결과:")
    calculate_metrics(true_labels_zs_2, pred_labels_zs_2, labels_2class, output_file=output_file, experiment_name='2분류 Zero-shot 평가 결과')

    print("\n2분류 실험: In-Context Learning 평가 결과:")
    calculate_metrics(true_labels_ic_2, pred_labels_ic_2, labels_2class, output_file=output_file, experiment_name='2분류 In-Context Learning 평가 결과')


    # **3분류 실험 (전체 레이블)**
    labels_3class = ['supports', 'refutes', 'not enough information']

    print(f"\n3분류 실험: Zero-shot 평가를 시작합니다...")
    true_labels_zs_3, pred_labels_zs_3 = zero_shot_evaluate(model, tokenizer, test_data, labels_3class, device)
    print("3분류 실험: Zero-shot 평가 완료.\n")

    print("3분류 실험: In-Context Learning 평가를 시작합니다...")
    true_labels_ic_3, pred_labels_ic_3 = in_context_evaluate(model, tokenizer, test_data, example_data, label_mapping, labels_3class, device)
    print("3분류 실험: In-Context Learning 평가 완료.\n")

    print("3분류 실험: Zero-shot 평가 결과:")
    calculate_metrics(true_labels_zs_3, pred_labels_zs_3, labels_3class, output_file=output_file, experiment_name='3분류 Zero-shot 평가 결과')

    print("\n3분류 실험: In-Context Learning 평가 결과:")
    calculate_metrics(true_labels_ic_3, pred_labels_ic_3, labels_3class, output_file=output_file, experiment_name='3분류 In-Context Learning 평가 결과')
