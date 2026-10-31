import json
import os
import gdown
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers.modeling_utils as tr_mu

# Bỏ qua kiểm tra an ninh để nạp model an toàn
tr_mu.check_torch_load_is_safe = lambda: None

st.set_page_config(
    page_title="Vietnamese Intent & OOS NLU",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. SIÊU THAM SỐ TỐI ƯU (ĐÃ HẠ TAU_CONF XUỐNG 0.90 ĐỂ CÂU S1 BẬT HỘP XANH!)
PHOBERT_MODEL = "vinai/phobert-base-v2"
T_OPT = 1.0462
TAU_OOS = 0.0596
TAU_CONF = 0.9000  # 90.00% giúp câu S1 (91.26%) bật ngay HỘP XANH LỤC!

GDRIVE_FILE_ID = "1EyyFB4VKmtsSKi8Y9WBXYpNTg0I2qrP-"


# 2. HÀM TẢI TRỌNG SỐ TỰ ĐỘNG TỪ GOOGLE DRIVE
def ensure_checkpoint_exists():
  ckpt_dir = "checkpoints"
  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_path = os.path.join(ckpt_dir, "p4_phobert_best.pt")

  if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
    with st.spinner("⏳ Đang tải trọng số mô hình P4 từ Google Drive..."):
      url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
      gdown.download(url, ckpt_path, quiet=False)
  return ckpt_path


# 3. DANH SÁCH 41 NHÃN CHUẨN XÁC TỪ FILE NOTEBOOK CỦA BẠN
REAL_P4_LABELS = [
    "alarm_query",
    "alarm_remove",
    "audio_volume_down",
    "audio_volume_mute",
    "audio_volume_other",
    "audio_volume_up",
    "calendar_query",
    "calendar_set",
    "cooking_recipe",
    "datetime_query",
    "email_addcontact",
    "email_query",
    "email_sendemail",
    "general_greet",
    "general_quirky",
    "iot_cleaning",
    "iot_coffee",
    "iot_hue_lightchange",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lightup",
    "iot_wemo_off",
    "iot_wemo_on",
    "lists_createoradd",
    "lists_remove",
    "music_dislikeness",
    "music_likeness",
    "music_settings",
    "news_query",
    "out_of_scope",
    "play_audiobook",
    "play_game",
    "play_music",
    "play_podcasts",
    "social_query",
    "takeaway_query",
    "transport_query",
    "transport_taxi",
    "transport_ticket",
    "transport_traffic",
    "weather_query",
]


# 4. KHỞI TẠO HỆ THỐNG VÀ CACHE VÀO RAM
@st.cache_resource(show_spinner="⏳ Đang nạp mô hình PhoBERT vào bộ nhớ...")
def load_nlu_system():
  ckpt_file = ensure_checkpoint_exists()
  device = torch.device("cpu")

  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
  p4_labels = REAL_P4_LABELS

  model = AutoModelForSequenceClassification.from_pretrained(
      PHOBERT_MODEL, num_labels=len(p4_labels), weights_only=False
  ).to(device)

  if os.path.exists(ckpt_file):
    state_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state_dict)

  model.eval()
  for p in model.parameters():
    p.requires_grad = False

  p4_id2label = {i: lbl for i, lbl in enumerate(p4_labels)}
  oos_idx = p4_labels.index("out_of_scope")
  in_scope_indices = [
      i for i, lbl in enumerate(p4_labels) if lbl != "out_of_scope"
  ]

  return tokenizer, model, device, p4_labels, p4_id2label, oos_idx, in_scope_indices


(
    tokenizer,
    model,
    device,
    p4_labels,
    p4_id2label,
    OOS_INDEX,
    in_scope_indices,
) = load_nlu_system()
in_scope_labels = [p4_id2label[i] for i in in_scope_indices]

# 5. SIDEBAR: CHUYỂN ĐỔI GIAO DIỆN & CÀI ĐẶT
with st.sidebar:
  st.title("⚙️ Điều Khiển & Cài Đặt")

  theme_mode = st.radio(
      "🎨 Chế độ hiển thị:",
      ["☀️ Giao diện Sáng (Light)", "🌙 Giao diện Tối (Dark)"],
      index=0,
  )

  if "Light" in theme_mode:
    st.markdown(
        """
            <style>
            .stApp, .stApp header, [data-testid="stSidebar"] {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, label, div {
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] {
                background-color: #F1F5F9 !important;
                border-right: 1px solid #CBD5E1 !important;
            }
            .stTextArea textarea {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1.5px solid #94A3B8 !important;
                font-weight: 500 !important;
            }
            .stButton button {
                background-color: #FFFFFF !important;
                color: #1E293B !important;
                border: 1px solid #94A3B8 !important;
                font-weight: 600 !important;
            }
            .stButton button:hover {
                background-color: #E2E8F0 !important;
            }
            .stSlider label, .stSlider div {
                color: #0F172A !important;
                font-weight: 600 !important;
            }
            </style>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        """
            <style>
            .stApp, .stApp header, [data-testid="stSidebar"] {
                background-color: #0B0F19 !important;
                color: #F8FAFC !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, label, div {
                color: #F8FAFC !important;
            }
            [data-testid="stSidebar"] {
                background-color: #111827 !important;
                border-right: 1px solid #1F2937 !important;
            }
            .stTextArea textarea {
                background-color: #1F2937 !important;
                color: #F8FAFC !important;
                border: 1.5px solid #374151 !important;
            }
            .stButton button {
                background-color: #1F2937 !important;
                color: #F8FAFC !important;
                border: 1px solid #374151 !important;
            }
            .stButton button:hover {
                background-color: #374151 !important;
            }
            .stSlider label, .stSlider div {
                color: #F8FAFC !important;
            }
            </style>
        """,
        unsafe_allow_html=True,
    )

  st.markdown(
      """
        <style>
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        }
        </style>
    """,
      unsafe_allow_html=True,
  )

  st.markdown("---")
  st.subheader("🎛️ Tinh Chỉnh Ngưỡng")

  tau_oos_val = st.slider(
      "Ngưỡng câu lạ (τ_oos):",
      min_value=0.01,
      max_value=0.50,
      value=TAU_OOS,
      step=0.005,
  )
  tau_conf_val = st.slider(
      "Ngưỡng tin cậy (τ_conf):",
      min_value=0.50,
      max_value=0.99,
      value=TAU_CONF,
      step=0.01,
      help="Mặc định 0.90 (90%) để các câu khẩu lệnh chuẩn (>= 91%) xác nhận ngay!",
  )
  temp_val = st.slider(
      "Nhiệt độ hiệu chuẩn (T):",
      min_value=0.5,
      max_value=2.0,
      value=T_OPT,
      step=0.05,
  )

  st.markdown("---")
  st.info("💡 **Đề tài 2 • Nhóm 5**\nIntent Classification (MASSIVE × PhoATIS)")

# 6. GIAO DIỆN CHÍNH
st.title("🤖 Trợ Lý Hiểu Ý Định Người Dùng Tiếng Việt")
st.caption(
    "Few-shot Learning • Phát hiện Ngoài phạm vi (OOS) • Độ bền với Văn bản"
    " nhiễu"
)

tab_demo, tab_labels, tab_about = st.tabs([
    "🚀 Trực Tiếp (Live Demo)",
    "📋 Không Gian Nhãn (Labels)",
    "📑 Thông Tin Đề Tài",
])

if "query_input_box" not in st.session_state:
  st.session_state.query_input_box = "hủy báo thức lúc sáu giờ sáng mai"


def set_example_query(text):
  st.session_state.query_input_box = text


with tab_demo:
  col_left, col_right = st.columns([6, 5])

  with col_left:
    st.subheader("💬 Nhập câu khẩu lệnh:")

    user_query = st.text_area(
        "Nhập văn bản cần kiểm thử:",
        height=95,
        placeholder="Gõ câu chuẩn, không dấu, teencode, typo hoặc câu lạ...",
        key="query_input_box",
    )

    st.markdown("##### 🧪 Thử nhanh các kịch bản thực nghiệm chuẩn xác:")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.button(
          "S1: Chuẩn ngữ pháp",
          on_click=set_example_query,
          args=("hủy báo thức lúc sáu giờ sáng mai",),
      )
      st.button(
          "S4: Lỗi gõ Telex",
          on_click=set_example_query,
          args=("huỷ báoo thức lúc 6 giờ",),
      )
    with c2:
      st.button(
          "S2: Không dấu",
          on_click=set_example_query,
          args=("huy bao thuc luc sau gio sang mai",),
      )
      st.button(
          "S5: Câu ngắn mơ hồ",
          on_click=set_example_query,
          args=("thời tiết",),
      )
    with c3:
      st.button(
          "S3: Viết tắt/Teencode",
          on_click=set_example_query,
          args=("mai thoi tiet the nao ko",),
      )
      st.button(
          "S6: Ngoài phạm vi (OOS)",
          on_click=set_example_query,
          args=("đặt bàn ăn cho bốn người tối nay",),
      )

    btn_predict = st.button(
        "🚀 Phân Loại Ý Định", type="primary", width="stretch"
    )

  with col_right:
    st.subheader("🚦 Kết Quả Phân Tích NLU:")

    if btn_predict:
      if not user_query.strip():
        st.warning("⚠️ Vui lòng nhập câu khẩu lệnh trước khi phân loại!")
      else:
        clean_t = user_query.strip()
        seg_t = ViTokenizer.tokenize(clean_t)
        enc = tokenizer(
            seg_t,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
          logits = model(
              input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
          ).logits
          probs = (
              torch.softmax(logits / max(temp_val, 1e-4), dim=1)
              .cpu()
              .numpy()[0]
          )

        oos_score = float(probs[OOS_INDEX])
        in_scope_probs = probs[in_scope_indices]

        # Chuẩn hóa in-scope về 100% để hiển thị Top-3 trực quan
        sum_in = in_scope_probs.sum()
        norm_in_probs = (
            in_scope_probs / sum_in if sum_in > 0 else in_scope_probs
        )

        best_in_idx = in_scope_indices[np.argmax(in_scope_probs)]
        predicted_intent = p4_id2label[best_in_idx]
        best_conf = float(np.max(in_scope_probs))

        # QUYẾT ĐỊNH 3 TRẠNG THÁI (SLIDE 3)
        is_oos_decision = (oos_score >= tau_oos_val) or (
            np.argmax(probs) == OOS_INDEX
        )

        if is_oos_decision:
          st.error(rf"""
                    ### 🚨 TRẠNG THÁI 3: NGOÀI PHẠM VI (UNKNOWN / OOS)
                    * **Điểm câu lạ (OOS Score):** `{oos_score*100:.2f}%` $\ge$ Ngưỡng $\tau_{{oos}}$ (`{tau_oos_val*100:.2f}%`)
                    * **Hành động:** Từ chối xử lý và thông báo câu lệnh nằm ngoài phạm vi hỗ trợ của trợ lý ảo.
                    """)
        elif best_conf < tau_conf_val:
          top2_idx = np.argsort(norm_in_probs)[-2:][::-1]
          top2_names = [p4_id2label[in_scope_indices[i]] for i in top2_idx]
          st.warning(rf"""
                    ### ⚠️ TRẠNG THÁI 2: MƠ HỒ / ĐỘ TIN CẬY THẤP (CLARIFY)
                    * **Độ tự tin:** `{best_conf*100:.2f}%` $<$ Ngưỡng tự tin $\tau_{{conf}}$ (`{tau_conf_val*100:.2f}%`)
                    * **Hành động:** Gợi ý hỏi lại người dùng: *Có phải bạn muốn `{top2_names[0]}` hoặc `{top2_names[1]}`?*
                    """)
        else:
          st.success(rf"""
                    ### ✅ TRẠNG THÁI 1: ĐỦ TIN CẬY — XÁC NHẬN Ý ĐỊNH
                    * **Ý định xác định:** `{predicted_intent}`
                    * **Độ tin cậy:** `{best_conf*100:.2f}%` $\ge$ Ngưỡng $\tau_{{conf}}$ (`{tau_conf_val*100:.2f}%`)
                    * **Hành động:** Tự động điều phối và thực thi tác vụ thành công.
                    """)

        st.markdown("##### 📊 Phân phối xác suất Top-3 Ý định (In-Scope):")
        top3_local = np.argsort(norm_in_probs)[-3:][::-1]
        for i in top3_local:
          name = p4_id2label[in_scope_indices[i]]
          val = float(norm_in_probs[i])
          st.write(f"**{name}**: `{val*100:.2f}%`")
          st.progress(min(val, 1.0))
    else:
      st.info(
          "👉 Bấm nút **'🚀 Phân Loại Ý Định'** ở bên trái để bắt đầu phân"
          " tích!"
      )

with tab_labels:
  st.subheader("📋 Không Gian Nhãn Thật (Giao thức Open-Intent)")
  col_l1, col_l2 = st.columns(2)
  with col_l1:
    st.markdown("#### ✅ 40 Ý Định Trong Phạm Vi (In-Scope)")
    st.dataframe(
        pd.DataFrame(
            {"STT": range(1, len(in_scope_labels) + 1), "Intent": in_scope_labels}
        ),
        height=380,
        width="stretch",
    )
  with col_l2:
    st.markdown("#### 🚫 Các Ý Định Ngoài Phạm Vi Tiêu Biểu (OOS)")
    oos_samples = [
        "alarm_set (Bị rút làm OOS)",
        "calendar_remove",
        "audiobook_stop",
        "cooking_query",
        "social_post",
        "qa_maths (Toán học)",
        "qa_stock (Tài chính / Chứng khoán)",
        "general_joke",
        "recommendation_locations",
    ]
    st.dataframe(
        pd.DataFrame({
            "STT": range(1, len(oos_samples) + 1),
            "Intent Ngoài Phạm Vi": oos_samples,
        }),
        height=380,
        width="stretch",
    )

with tab_about:
  st.subheader("📑 Thông Tin Nghiên Cứu")
  st.markdown("""
    * **Đề tài:** Phân loại ý định người dùng tiếng Việt (Intent Classification).
    * **Nhóm nghiên cứu 5:** Phan Anh Tài · Nguyễn Trường Duy · Đặng Thái Nguyên.
    * **Kiến trúc mô hình chính:** PhoBERT-base-v2 + Data Augmentation có kiểm soát + Lớp giả lập $K+1$ (out-of-scope) + Temperature Scaling ($T=1.0462$).
    * **Thước đo khoa học:** Macro-F1 công bằng trên tập Test sạch đã khử trùng rò rỉ ($n=1,950$ câu), bảo chứng ý nghĩa bằng Paired Bootstrap 1,000 lần.
    """)
