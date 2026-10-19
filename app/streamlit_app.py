import os
import zipfile
import streamlit as st
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

MODEL_NAME = "John-asd123/vietnamese-intent-p4"
LOCAL_DIR = "./model_extracted"

st.set_page_config(
    page_title="Vietnamese Intent & OOS Detection",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Hệ Thống Phân Loại Ý Định & Phát Hiện Ngoài Miền (OOS)")
st.markdown("**Mô hình:** PhoBERT-v2 (Fine-tuned P4) | **Số lượng nhãn:** 41 Intent")

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Kiểm tra xem đã giải nén sẵn chưa, nếu chưa thì tải file zip từ Hugging Face
    if not os.path.exists(LOCAL_DIR) or not os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
        os.makedirs(LOCAL_DIR, exist_ok=True)
        with st.spinner("📥 Đang tải gói mô hình từ Hugging Face (chỉ mất 1 lần duy nhất)..."):
            zip_path = hf_hub_download(repo_id=MODEL_NAME, filename="model_p4.zip")
            
        with st.spinner("📦 Đang giải nén mô hình..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_DIR)
                
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    
    model.to(device)
    model.eval()
    return tokenizer, model, device

try:
    tokenizer, model, device = load_model_and_tokenizer()
    has_model = True
    st.success("✅ Tải mô hình và trọng số thành công từ gói nén!")
except Exception as e:
    st.error(f"⚠️ Lỗi khi tải mô hình: {e}")
    has_model = False

# Sidebar cấu hình ngưỡng OOS
st.sidebar.header("⚙️ Cấu hình hệ thống")
tau_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy an toàn (Threshold τ):",
    min_value=0.0, max_value=1.0, value=0.70, step=0.01,
    help="Nếu xác suất cao nhất < τ hoặc nhãn là out_of_scope, hệ thống sẽ cảnh báo Ngoài miền."
)

# Giao diện nhập liệu
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Nhập câu lệnh")
    example_selected = st.selectbox(
        "💡 Chọn câu ví dụ mẫu:",
        [
            "-- Tự nhập câu lệnh --",
            "đặt báo thức lúc sáu giờ sáng mai",
            "bật cho tôi bài hát Lạc Trôi của Sơn Tùng",
            "xem lịch họp buổi chiều nay có những gì",
            "tat den phong khach giup toi",
            "mai có mưa không em",
            "cổ phiếu Vinfast hôm nay giá bao nhiêu tiền"
        ]
    )
    
    default_text = "" if example_selected == "-- Tự nhập câu lệnh --" else example_selected
    input_text = st.text_area("Nội dung câu lệnh:", value=default_text, height=100)
    btn_predict = st.button("⚡ Phân tích ý định", type="primary")

with col2:
    st.subheader("📊 Kết quả phân tích")
    
    if btn_predict and input_text.strip():
        if has_model:
            segmented_text = ViTokenizer.tokenize(input_text.strip())
            
            inputs = tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            id2label = getattr(model.config, "id2label", None)
            if id2label:
                prob_dict = {
                    id2label.get(str(i), id2label.get(i, f"class_{i}")): float(probs[i])
                    for i in range(len(probs))
                }
            else:
                prob_dict = {f"class_{i}": float(probs[i]) for i in range(len(probs))}
                
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top_intent, max_p = sorted_probs[0]
            
            is_oos = (max_p < tau_threshold) or (top_intent == "out_of_scope")
            
            if is_oos:
                st.error(f"🚨 **CẢNH BÁO: CÂU LỆNH NGOÀI MIỀN (OUT-OF-SCOPE)**")
                st.info(f"- Độ tin cậy tối đa: `{max_p:.2%}` < Ngưỡng an toàn τ (`{tau_threshold:.2%}`)")
            else:
                st.success(f"✅ **XÁC NHẬN: Ý ĐỊNH HỢP LỆ (IN-SCOPE)**")
                st.markdown(f"- Ý định dự đoán: **{top_intent}**")
                st.markdown(f"- Độ tin cậy: `{max_p:.2%}`")
                
            st.markdown("---")
            st.write("**Top-5 Xác suất ý định:**")
            for intent_name, p_val in sorted_probs[:5]:
                st.progress(float(p_val), text=f"{intent_name}: {p_val:.2%}")
        else:
            st.warning("Mô hình chưa sẵn sàng để dự đoán.")
