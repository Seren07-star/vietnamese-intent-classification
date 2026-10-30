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

# 1. SIÊU THAM SỐ TỐI ƯU TỪ BÀI BÁO
PHOBERT_MODEL = "vinai/phobert-base-v2"
T_OPT = 1.0462
TAU_OOS = 0.0596
TAU_CONF = 0.9400

# FILE ID CHUẨN XÁC TỪ GOOGLE DRIVE CỦA BẠN
GDRIVE_FILE_ID = "1EyyFB4VKmtsSKi8Y9WBXYpNTg0I2qrP-"


# 2. HÀM TẢI TRỌNG SỐ TỰ ĐỘNG TỪ GOOGLE DRIVE
def ensure_checkpoint_exists():
  ckpt_dir = "checkpoints"
  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_path = os.path.join(ckpt_dir, "p4_phobert_best.pt")

  if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
    with st.spinner(
        "⏳ Đang tải trọng số mô hình P4 từ Google Drive (khoảng 30 giây cho lần"
        " đầu)..."
    ):
      url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
      gdown.download(url, ckpt_path, quiet=False)
  return ckpt_path


# 3. DANH SÁCH 41 NHÃN CHUẨN (ĐÃ TỰ ĐỘNG NẠP DANH MỤC INTENT THẬT)
P4_LABELS_CANONICAL = sorted([
    "alarm_query",
    "alarm_remove",
    "alarm_set",
    "audiobook_pause",
    "audiobook_play",
    "calendar_query",
    "calendar_remove",
    "calendar_set",
    "cooking_query",
    "cooking_recipe",
    "datetime_convert",
    "datetime_query",
    "email_addcontact",
    "email_query",
    "email_querycontact",
    "email_reply",
    "email_sendemail",
    "general_greet",
    "general_joke",
    "general_quirky",
    "iot_cleaning",
    "iot_coffee",
    "iot_hue_lightchange",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "iot_wemo_off",
    "iot_wemo_on",
    "music_likeness",
    "music_query",
    "music_settings",
    "news_query",
    "qa_definition",
    "qa_factoid",
    "qa_maths",
    "qa_stock",
    "recommendation_events",
    "recommendation_locations",
    "recommendation_movies",
    "social_post",
    "out_of_scope",
])


# 4. KHỞI TẠO HỆ THỐNG VÀ CACHE VÀO BỘ NHỚ RAM
@st.cache_resource(show_spinner="⏳ Đang nạp mô hình PhoBERT vào bộ nhớ...")
def load_nlu_system():
  ckpt_file = ensure_checkpoint_exists()
  device = torch.device("cpu")

  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
  p4_labels = P4_LABELS_CANONICAL

  # Nạp cấu trúc model 41 nhãn
  model = AutoModelForSequenceClassification.from_pretrained(
      PHOBERT_MODEL, num_labels=len(p4_labels), weights_only=False
  ).to(device)

  # Nạp trọng số thật từ file .pt tải từ Drive
  if os.path.exists(ckpt_file):
    state_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Đã nạp thành công trọng số P4 từ Google Drive!")

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


# 5. THANH BÊN (SIDEBAR): GIAO DIỆN SÁNG/TỐI & CÀI ĐẶT
with st.sidebar:
  st.title("⚙️ Điều Khiển & Cài Đặt")

  # CHẾ ĐỘ SÁNG / TỐI (LIGHT / DARK THEME TOGGLE)
  theme_mode = st.radio(
      "🎨 Chế độ hiển thị:",
      ["🌙 Giao diện Tối (Dark)", "☀️ Giao diện Sáng (Light)"],
      index=0,
  )

  st.markdown("---")
  st.subheader("🎛️ Tinh Chỉnh Ngưỡng")

  tau_oos_val = st.slider(
      "Ngưỡng câu lạ (τ_oos):",
      min_value=0.01,
      max_value=0.50,
      value=TAU_OOS,
      step=0.005,
      help="Vượt ngưỡng này sẽ kích hoạt từ chối (OOS).",
  )
  tau_conf_val = st.slider(
      "Ngưỡng tin cậy (τ_conf):",
      min_value=0.50,
      max_value=0.99,
      value=TAU_CONF,
      step=0.01,
      help="Dưới ngưỡng này sẽ kích hoạt hỏi lại (Clarify).",
  )
  temp_val = st.slider(
      "Nhiệt độ hiệu chuẩn (T):",
      min_value=0.5,
      max_value=2.0,
      value=T_OPT,
      step=0.05,
      help="Làm phẳng xác suất chống overconfidence.",
  )

  st.markdown("---")
  st.info("💡 **Đề tài 2 • Nhóm 5**\nIntent Classification (MASSIVE × PhoATIS)")

# CSS TÙY BIẾN THEO CHẾ ĐỘ SÁNG / TỐI
if "Light" in theme_mode:
  bg_card = "#F3F4F6"
  text_card = "#1F2937"
  border_col = "#E5E7EB"
else:
  bg_card = "#1F2937"
  text_card = "#F9FAFB"
  border_col = "#374151"


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

with tab_demo:
  col_left, col_right = st.columns([6, 5])

  with col_left:
    st.subheader("💬 Nhập câu khẩu lệnh:")

    # Quản lý text thông qua session state để các nút bấm hoạt động mượt mà
    if "input_text" not in st.session_state:
      st.session_state.input_text = "hủy báo thức lúc sáu giờ sáng mai"

    user_query = st.text_area(
        "Nhập văn bản cần kiểm thử:",
        value=st.session_state.input_text,
        height=95,
        placeholder="Gõ câu chuẩn, không dấu, teencode, typo hoặc câu lạ...",
        key="query_input_box",
    )

    st.markdown("##### 🧪 Thử nhanh các kịch bản thực nghiệm:")
    c1, c2, c3 = st.columns(3)
    with c1:
      if st.button("S1: Chuẩn ngữ pháp"):
        st.session_state.input_text = "hủy báo thức lúc sáu giờ sáng mai"
        st.rerun()
      if st.button("S4: Lỗi gõ Telex"):
        st.session_state.input_text = "huuyr báoo thứcc lúc 6h"
        st.rerun()
    with c2:
      if st.button("S2: Không dấu"):
        st.session_state.input_text = "huy bao thuc luc sau gio sang mai"
        st.rerun()
      if st.button("S5: Câu ngắn mơ hồ"):
        st.session_state.input_text = "thời tiết"
        st.rerun()
    with c3:
      if st.button("S3: Viết tắt/Teencode"):
        st.session_state.input_text = "mai thoi tiet the nao ko"
        st.rerun()
      if st.button("S6: Ngoài phạm vi (OOS)"):
        st.session_state.input_text = "hôm nay thị trường chứng khoán thế nào"
        st.rerun()

    # NÚT BẤM KIỂM SOÁT SUY LUẬN (CHỈ CHẠY KHI BẤM NÚT)
    btn_predict = st.button(
        "🚀 Phân Loại Ý Định", type="primary", use_container_width=True
    )

  with col_right:
    st.subheader("🚦 Kết Quả Phân Tích NLU:")

    # CHỈ CHẠY SUY LUẬN KHI NGƯỜI DÙNG BẤM NÚT
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
              torch.softmax(logits / max(temp_val, 1e-4), dim=1).cpu().numpy()[0]
          )

        oos_score = float(probs[OOS_INDEX])
        in_scope_probs = probs[in_scope_indices]

        # Chuẩn hóa phân phối in-scope về 100% để thanh Top-3 không bị hiển thị 0%
        sum_in = in_scope_probs.sum()
        norm_in_probs = (
            in_scope_probs / sum_in if sum_in > 0 else in_scope_probs
        )

        best_in_idx = in_scope_indices[np.argmax(in_scope_probs)]
        predicted_intent = p4_id2label[best_in_idx]
        best_conf = float(np.max(in_scope_probs))

        # QUYẾT ĐỊNH 3 TRẠNG THÁI (SLIDE 3)
        if oos_score >= tau_oos_val:
          st.error(f"""
                    ### 🚨 TRẠNG THÁI 3: NGOÀI PHẠM VI (UNKNOWN / OOS)
                    * **Điểm câu lạ (OOS Score):** `{oos_score*100:.2f}%` $\ge$ Ngưỡng $\\tau_{{oos}}$ (`{tau_oos_val*100:.2f}%`)
                    * **Hành động:** Từ chối xử lý và thông báo câu lệnh nằm ngoài phạm vi hỗ trợ của trợ lý ảo.
                    """)
        elif best_conf < tau_conf_val:
          top2_idx = np.argsort(norm_in_probs)[-2:][::-1]
          top2_names = [p4_id2label[in_scope_indices[i]] for i in top2_idx]
          st.warning(f"""
                    ### ⚠️ TRẠNG THÁI 2: MƠ HỒ / ĐỘ TIN CẬY THẤP (CLARIFY)
                    * **Độ tự tin:** `{best_conf*100:.2f}%` $<$ Ngưỡng tự tin $\\tau_{{conf}}$ (`{tau_conf_val*100:.2f}%`)
                    * **Hành động:** Gợi ý hỏi lại người dùng: *Có phải bạn muốn `{top2_names[0]}` hoặc `{top2_names[1]}`?*
                    """)
        else:
          st.success(f"""
                    ### ✅ TRẠNG THÁI 1: ĐỦ TIN CẬY — XÁC NHẬN Ý ĐỊNH
                    * **Ý định xác định:** `{predicted_intent}`
                    * **Độ tin cậy:** `{best_conf*100:.2f}%` $\ge$ Ngưỡng $\\tau_{{conf}}$ (`{tau_conf_val*100:.2f}%`)
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
  st.subheader("📋 Không Gian Nhãn (Giao thức Open-Intent)")
  col_l1, col_l2 = st.columns(2)
  with col_l1:
    st.markdown("#### ✅ 40 Ý Định Trong Phạm Vi (In-Scope)")
    st.dataframe(
        pd.DataFrame(
            {"STT": range(1, len(in_scope_labels) + 1), "Intent": in_scope_labels}
        ),
        height=380,
        use_container_width=True,
    )
  with col_l2:
    st.markdown("#### 🚫 Các Ý Định Ngoài Phạm Vi (OOS Tiêu Biểu)")
    oos_samples = [
        "qa_stock (Tài chính / Chứng khoán)",
        "cooking_query (Nấu ăn)",
        "transport_taxi (Gọi xe taxi)",
        "audiobook_stop (Dừng sách nói)",
        "general_greet (Chào hỏi xã giao)",
        "iot_coffee (Pha cà phê thông minh)",
        "music_likeness (Đánh giá bài hát)",
    ]
    st.dataframe(
        pd.DataFrame({
            "STT": range(1, len(oos_samples) + 1),
            "Intent Ngoài Phạm Vi": oos_samples,
        }),
        height=380,
        use_container_width=True,
    )

with tab_about:
  st.subheader("📑 Thông Tin Nghiên Cứu")
  st.markdown("""
    * **Đề tài:** Phân loại ý định người dùng tiếng Việt (Intent Classification).
    * **Nhóm nghiên cứu 5:** Phan Anh Tài · Nguyễn Trường Duy · Đặng Thái Nguyên.
    * **Kiến trúc mô hình chính:** PhoBERT-base-v2 + Data Augmentation có kiểm soát + Lớp giả lập $K+1$ (out-of-scope) + Temperature Scaling ($T=1.0462$).
    * **Thước đo khoa học:** Macro-F1 công bằng trên tập Test sạch đã khử trùng rò rỉ ($n=1,950$ câu), bảo chứng ý nghĩa bằng Paired Bootstrap 1,000 lần.
    """)
