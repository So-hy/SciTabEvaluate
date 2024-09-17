# main.py

import torch
from data_utils import load_data, split_data, linearize_table
from model_utils import load_model, generate_prediction
from evaluation import calculate_metrics

def map_prediction_to_label(prediction):
    """모델의 출력 문자열을 레이블로 매핑합니다."""
    prediction = prediction.lower()
    if 'true' in prediction or 'a' in prediction:
        return 'supports'
    elif 'false' in prediction or 'b' in prediction:
        return 'refutes'
    elif 'unknown' in prediction or 'c' in prediction:
        return 'not enough information'
    else:
        return 'unknown'

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

    # 레이블 맵핑
    label_mapping = {
        'supports': 'True',
        'refutes': 'False',
        'not enough information': 'Unknown'
    }

    # 결과를 저장할 파일 열기
    output_file = 'evaluation_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        # 2-class 실험
        print("=== 2-class 실험 (supports, refutes) ===")
        f.write("=== 2-class 실험 (supports, refutes) ===\n")
        data_2class = [sample for sample in data if sample['label'] != 'not enough information']
        example_data_2class, test_data_2class = split_data(data_2class)

        # 클래스 레이블 설정
        class_labels_2class = ['supports', 'refutes']

        print(f"예시 데이터 개수 (2-class): {len(example_data_2class)}")
        print(f"테스트 데이터 개수 (2-class): {len(test_data_2class)}")
        f.write(f"예시 데이터 개수 (2-class): {len(example_data_2class)}\n")
        f.write(f"테스트 데이터 개수 (2-class): {len(test_data_2class)}\n")

        print("Zero-shot 평가를 시작합니다...")
        true_labels_zs_2class, pred_labels_zs_2class = zero_shot_evaluate(model, tokenizer, test_data_2class, device)
        print("Zero-shot 평가 완료.\n")

        print("In-Context Learning 평가를 시작합니다...")
        true_labels_ic_2class, pred_labels_ic_2class = in_context_evaluate(model, tokenizer, test_data_2class, example_data_2class, label_mapping, device)
        print("In-Context Learning 평가 완료.\n")

        print("Zero-shot 평가 결과 (2-class):")
        f.write("\nZero-shot 평가 결과 (2-class):\n")
        calculate_metrics(true_labels_zs_2class, pred_labels_zs_2class, class_labels_2class, file=f)
        f.write("\n")

        print("\nIn-Context Learning 평가 결과 (2-class):")
        f.write("\nIn-Context Learning 평가 결과 (2-class):\n")
        calculate_metrics(true_labels_ic_2class, pred_labels_ic_2class, class_labels_2class, file=f)
        f.write("\n")

        # 3-class 실험
        print("\n=== 3-class 실험 (supports, refutes, not enough information) ===")
        f.write("\n=== 3-class 실험 (supports, refutes, not enough information) ===\n")
        data_3class = data.copy()
        example_data_3class, test_data_3class = split_data(data_3class)

        # 클래스 레이블 설정
        class_labels_3class = ['supports', 'refutes', 'not enough information']

        print(f"예시 데이터 개수 (3-class): {len(example_data_3class)}")
        print(f"테스트 데이터 개수 (3-class): {len(test_data_3class)}")
        f.write(f"예시 데이터 개수 (3-class): {len(example_data_3class)}\n")
        f.write(f"테스트 데이터 개수 (3-class): {len(test_data_3class)}\n")

        print("Zero-shot 평가를 시작합니다...")
        true_labels_zs_3class, pred_labels_zs_3class = zero_shot_evaluate(model, tokenizer, test_data_3class, device)
        print("Zero-shot 평가 완료.\n")

        print("In-Context Learning 평가를 시작합니다...")
        true_labels_ic_3class, pred_labels_ic_3class = in_context_evaluate(model, tokenizer, test_data_3class, example_data_3class, label_mapping, device)
        print("In-Context Learning 평가 완료.\n")

        print("Zero-shot 평가 결과 (3-class):")
        f.write("\nZero-shot 평가 결과 (3-class):\n")
        calculate_metrics(true_labels_zs_3class, pred_labels_zs_3class, class_labels_3class, file=f)
        f.write("\n")

        print("\nIn-Context Learning 평가 결과 (3-class):")
        f.write("\nIn-Context Learning 평가 결과 (3-class):\n")
        calculate_metrics(true_labels_ic_3class, pred_labels_ic_3class, class_labels_3class, file=f)
