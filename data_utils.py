# data_utils.py

import json
from random import shuffle

def load_data(data_file):
    """SCITAB Load."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def split_data(data, num_examples=5, num_claims_per_table=5):
    """Split Data."""
    shuffle(data)

    example_tables = {}
    example_data = []
    test_data = []

    for sample in data:
        table_id = sample['table_id']
        if len(example_tables) < num_examples:
            if table_id not in example_tables:
                example_tables[table_id] = True
                example_data.append(sample)
            elif len([s for s in example_data if s['table_id'] == table_id]) < num_claims_per_table:
                example_data.append(sample)
            else:
                test_data.append(sample)
        else:
            test_data.append(sample)
    return example_data, test_data

def linearize_table(sample):
    """Linearize Table..."""
    # 표 캡션
    caption = sample['table_caption']

    # 표 선형화
    table_str = ''
    headers = sample['table_column_names']
    rows = sample['table_content_values']

    for row in rows:
        row_str = ' | '.join([f"{headers[i]}: {row[i]}" for i in range(len(headers))])
        table_str += row_str + '\n'

    # 주장
    claim = sample['claim']

    # 입력 구성
    input_text = f"""Table Caption:
{caption}

Table:
{table_str}

Claim:
{claim}

Based on the information in the table, is the above claim true?
A) True
B) False
C) Unknown?
Answer:"""

    return input_text
