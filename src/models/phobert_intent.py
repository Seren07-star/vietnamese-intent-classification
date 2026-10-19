import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class IntentClassificationModel:
    """
    Wrapper cho mô hình PhoBERT-base-v2 phục vụ bài toán 
    Phân loại ý định (Intent Classification) và Phát hiện ngoài phạm vi (OOS Detection) với đầu ra K+1.
    """
    def __init__(self, model_name="vinai/phobert-base-v2", num_labels=41):
        self.model_name = model_name
        self.num_labels = num_labels
        
    def build_model(self, id2label=None, label2id=None):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id
        )
        return model
