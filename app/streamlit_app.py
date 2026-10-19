import streamlit as st
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Tên Repository chính thức trên Hugging Face
MODEL_NAME = "John-asd123/vietnamese-intent-p4"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Tải chính xác file trọng số 1.11 GB trực tiếp qua Hugging Face Hub Client (Bỏ qua Git LFS pointer)
    # Dùng hf_hub_download để đảm bảo tải đủ 1.11 GB thực tế
    model_weight_path = hf_hub_download(repo_id=MODEL_NAME, filename="model.safetensors")
    config_path = hf_hub_download(repo_id=MODEL_NAME, filename="config.json")
    
    # Load model trực tiếp từ repo Hugging Face
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    )
    
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Tải mô hình
try:
    with st.spinner("Đang tải mô hình từ Hugging Face... Vui lòng đợi trong giây lát..."):
        tokenizer, model, device = load_model()
    has_model = True
    st.success("✅ Đã tải mô hình thành công!")
except Exception as e:
    st.error(f"⚠️ Chưa tải được mô hình: {e}")
    has_model = False

# Sidebar cấu hình ngưỡng
st.sidebar.header("⚙️ Cấu hình hệ thống")
tau_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy an toàn (Threshold τ):",
    min_value=0.0, max_value=1.0, value=0.70, step=0.01,
    help="Nếu xác suất cao nhất < τ hoặc nhãn dự đoán là out_of_scope, hệ thống sẽ cảnh báo Ngoài miền (OOS)."
)

# Giao diện chính
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Nhập câu lệnh")
    example_selected = st.selectbox(
        "💡 Chọn câu ví dụ mẫu để test nhanh:",
        [
            "-- Tự nhập câu lệnh --",
            "đặt báo thức lúc sáu giờ sáng mai",
            "bật cho tôi bài hát Lạc Trôi của Sơn Tùng",
            "xem lịch họp buổi chiều nay có những gì",
            "tat den phong khach giup toi",
            "mai có mưa không em",
            "cổ phiếu Vinfast hôm nay giá bao nhiêu tiền",
            "hôm nay ăn gì ngon bổ rẻ"
        ]
    )
    
    default_text = "" if example_selected == "-- Tự nhập câu lệnh --" else example_selected
    input_text = st.text_area("Nội dung câu lệnh:", value=default_text, height=100, placeholder="Gõ câu lệnh tại đây...")
    btn_predict = st.button("⚡ Phân tích ý định", type="primary")

with col2:
    st.subheader("📊 Kết quả phân tích")
    
    if btn_predict and input_text.strip():
        # Tách từ tiếng Việt qua PyVi
        segmented_text = ViTokenizer.tokenize(input_text.strip())
        
        if has_model:
            inputs = tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            # Trích xuất id2label từ config của mô hình
            id2label = getattr(model.config, "id2label", None)
            if id2label:
                # Ép kiểu index về str/int để lấy đúng nhãn
                prob_dict = {
                    id2label.get(str(i), id2label.get(i, f"class_{i}")): float(probs[i])
                    for i in range(len(probs))
                }
            else:
                prob_dict = {DEFAULT_LABELS[i]: float(probs[i]) for i in range(min(len(probs), len(DEFAULT_LABELS)))}
        else:
            prob_dict = {label: 1.0 / len(DEFAULT_LABELS) for label in DEFAULT_LABELS}
            
        # Sắp xếp các intent theo xác suất giảm dần
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_intent, max_p = sorted_probs[0]
        
        # Kiểm tra điều kiện OOS
        is_oos = (max_p < tau_threshold) or (top_intent == "out_of_scope")
        
        if is_oos:
            st.error(f"🚨 **CẢNH BÁO: CÂU LỆNH NGOÀI MIỀN (OUT-OF-SCOPE / UNKNOWN)**")
            st.info(f"- Độ tin cậy tối đa: `{max_p:.2%}` < Ngưỡng an toàn τ (`{tau_threshold:.2%}`)")
        else:
            st.success(f"✅ **XÁC NHẬN: Ý ĐỊNH HỢP LỆ (IN-SCOPE)**")
            st.markdown(f"- Ý định: **{top_intent}**")
            st.markdown(f"- Độ tin cậy: `{max_p:.2%}`")
            
        st.markdown("---")
        st.write("**Top-5 Xác suất ý định:**")
        for intent_name, p_val in sorted_probs[:5]:
            st.progress(float(p_val), text=f"{intent_name}: {p_val:.2%}")
