import streamlit as st
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = [
    "alarm_query", "audio_volume_down", "audio_volume_mute", "audio_volume_other",
    "calendar_remove", "calendar_set", "cooking_query", "cooking_recipe",
    "datetime_clock", "datetime_convert", "email_addcontact", "email_query",
    "email_querycontact", "email_sendemail", "general_affirm", "general_commandstop",
    "general_confirm", "general_dontunderstand", "general_explain", "general_joke",
    "general_negate", "general_praise", "general_quirky", "iot_cleaning",
    "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
    "iot_hue_lighton", "iot_wemo_off", "iot_wemo_on", "music_likeness",
    "music_query", "news_query", "play_audiobook", "play_game",
    "play_music", "play_podcasts", "play_radio", "qa_definition", "out_of_scope"
]

# 👉 Trỏ trực tiếp tới repo mô hình của bạn trên Hugging Face
MODEL_NAME = "John-asd123/vietnamese-intent-p4"

st.set_page_config(
    page_title="Vietnamese Intent & OOS Detection",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Hệ Thống Phân Loại Ý Định & Phát Hiện Ngoài Miền (OOS) Tiếng Việt")
st.markdown("**Mô hình:** PhoBERT-v2 (Fine-tuned P4) | **Accuracy:** 88.34%")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hugging Face tự động tải trọng số đã train về cache của Streamlit Cloud cực kỳ nhanh
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device

try:
    with st.spinner("Đang tải mô hình từ Hugging Face... Vui lòng đợi trong giây lát..."):
        tokenizer, model, device = load_model()
    has_model = True
    st.success("✅ Đã tải mô hình thành công!")
except Exception as e:
    st.warning(f"⚠️ Chưa tải được mô hình: {e}")
    has_model = False

# Sidebar cấu hình ngưỡng
st.sidebar.header("⚙️ Cấu hình hệ thống")
tau_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy an toàn (Threshold τ):",
    min_value=0.0, max_value=1.0, value=0.70, step=0.01,
    help="Nếu xác suất cao nhất < τ hoặc là out_of_scope, hệ thống sẽ cảnh báo Ngoài miền (OOS)."
)

# Giao diện chính
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Nhập câu lệnh")
    example_selected = st.selectbox(
        "💡 Chọn câu ví dụ mẫu để test nhanh:",
        [
            "-- Tự nhập câu lệnh --",
            "bật cho tôi bài hát Lạc Trôi của Sơn Tùng",
            "đặt báo thức lúc sáu giờ sáng mai",
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
        segmented_text = ViTokenizer.tokenize(input_text.strip())
        
        if has_model:
            inputs = tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        else:
            probs = [1.0 / len(LABELS)] * len(LABELS)
            
        prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_intent, max_p = sorted_probs[0]
        
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
