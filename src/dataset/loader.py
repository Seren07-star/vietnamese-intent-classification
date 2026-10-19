import json
import pandas as pd

def load_jsonl(file_path):
    """Hàm chuẩn để nạp dữ liệu định dạng JSONL cho MASSIVE & PhoATIS"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def get_label_mappings(train_file_path):
    """Tái thiết lập ánh xạ nhãn từ tập train"""
    df = load_jsonl(train_file_path)
    labels = sorted(list(df['intent'].unique()))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label
