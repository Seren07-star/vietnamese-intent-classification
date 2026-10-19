import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Tên Repository trên Hugging Face
MODEL_NAME = "John-asd123/vietnamese-intent-p4"

st.set_page_config(
    page_title="Vietnamese Intent & OOS Detection",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Hệ Thống Phân Loại Ý Định & Phát Hiện Ngoài Miền (OOS) Tiếng Việt")
st.markdown("**Mô hình:** PhoBERT-v2 (Fine-tuned P4) | **Accuracy:** 88.34%")

# Định nghĩa cấu trúc mô hình chính xác như lúc train trên Kaggle
class PhoBertForIntentClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.phobert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy vector đại diện [CLS]
        cls_output = outputs[0][:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải tokenizer và config từ Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # 2. Khởi tạo mô hình theo đúng khung kiến trúc
    model = PhoBertForIntentClassification(config)
    
    # 3. Tải file trọng số thật (.safetensors) trực tiếp từ HF Hub
    weights_path = hf_hub_download(repo_id=MODEL_NAME, filename="model.safetensors")
    state_dict = load_file(weights_path)
    
    # 4. Map trọng số vào mô hình (strict=False để tránh lỗi mismatch tên prefix)
    missing, unexpected
