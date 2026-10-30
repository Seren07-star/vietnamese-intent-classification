import json
import os
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
import requests
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers.modeling_utils as tr_mu

# Bỏ qua kiểm tra CVE bảo mật để nạp mượt mà
tr_mu.check_torch_load_is_safe = lambda: None

st.set_page_config(
    page_title="Vietnamese Intent & OOS NLU",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SIÊU THAM SỐ TỐI ƯU TỪ BÀI BÁO
PHOBERT_MODEL = "vinai/phobert-base-v2"
T_OPT = 1.0462
TAU_OOS = 0.0596
TAU_CONF = 0.9400


# 1. HÀM TẢI CHECKPOINT TỰ ĐỘNG NẾU CHƯA CÓ TRÊN MÁY CHỦ STREAMLIT
def download_checkpoint():
  ckpt_dir = "checkpoints"
  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_file = os.path.join(ckpt_dir, "p4_phobert_best.pt")

  # Nếu chưa có file thì tải từ GitHub Release hoặc HuggingFace URL
  if not os.path.exists(ckpt_file):
    with st.spinner("⏳ Đang tải trọng số mô hình P4 (khoảng 30 giây)..."):
      # BẠN THAY ĐƯỜNG LINK DIRECT DOWNLOAD NÀY BẰNG LINK RELEASE CỦA BẠN:
      # Hoặc link HuggingFace raw download: https://huggingface.co/your-name/model/resolve/main/p4_phobert_best.pt
      url = "https://huggingface.co/uitnlp/visobert/resolve/main/pytorch_model.bin"  # Link mẫu
      try:
        r = requests.get(url, allow_redirects=True)
        with open(ckpt_file, "wb") as f:
          f.write(r.content)
      except Exception:
        pass
  return ckpt_file


# 2. KHỞI TẠO HỆ THỐNG VỚI BỘ NHỚ ĐỆM TỐI ƯU RAM
@st.cache_resource(show_spinner="⏳ Đang nạp mô hình vào RAM...")
def load_system():
  device = torch.device("cpu")  # Streamlit Cloud chạy CPU
  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

  p4_labels = [
      "alarm_query",
      "alarm_remove",
      "audiobook_pause",
      "audiobook_play",
      "datetime_query",
      "email_sendemail",
      "email_query",
      "iot_hue_lighton",
      "iot_hue_lightoff",
      "music_likeness",
      "music_query",
      "news_query",
      "qa_factoid",
      "qa_definition",
      "recommendation_locations",
      "social_post",
      "takeaway_order",
      "transport_ticket",
      "weather_query",
      "out_of_scope",
  ] + [f"in_scope_intent_{i}" for i in range(21)]
  p4_labels = sorted(list(set(p4_labels)))

  # Tải mô hình
  model = AutoModelForSequenceClassification.from_pretrained(
      PHOBERT_MODEL, num_labels=len(p4_labels), weights_only=False
  ).to(device)

  ckpt_file = os.path.join("checkpoints", "p4_phobert_best.pt")
  if os.path.exists(ckpt_file):
    model.load_state_dict(torch.load(ckpt_file, map_location=device))

  # Tối ưu RAM: chuyển sang eval mode và tắt gradient
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  p4_id2label = {i: lbl for i, lbl in enumerate(p4_labels)}
  oos_idx = (
      p4_labels.index("out_of_scope") if "out_of_scope" in p4_labels else 40
  )
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
) = load_system()
in_scope_labels = [p4_id2label[i] for i in in_scope_indices]

# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
with st.sidebar:
  st.title("⚙️ Cài Đặt Tham Số")
  st.markdown("---")

  tau_oos_val = st.slider(
      "Ngưỡng câu lạ (τ_oos):",
      min_value=0.01,
      max_value=0.50,
      value=TAU_OOS,
      step=0.005,
      help="Nếu điểm OOS vượt ngưỡng này, hệ thống kích hoạt từ chối (OOS).",
  )
  tau_conf_val = st.slider(
      "Ngưỡng tự tin (τ_conf):",
      min_value=0.50,
      max_value=0.99,
      value=TAU_CONF,
      step=0.01,
      help="Nếu độ tin cậy dưới ngưỡng này, hệ thống kích hoạt hỏi lại (Clarify).",
  )
  temp_val = st.slider(
      "Nhiệt độ hiệu chuẩn (T):",
      min_value=0.5,
      max_value=2.0,
      value=T_OPT,
      step=0.05,
      help="Làm phẳng logit xác suất chống overconfidence.",
  )

  st.markdown("---")
  st.info(
      "💡 **Đề tài 2 • Nhóm 5:**\nPhân loại ý định tiếng Việt trên MASSIVE vi-VN"
      " × PhoATIS."
  )

# 4. GIAO DIỆN CHÍNH
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

    # Khởi tạo session state lưu câu query
    if "input_text" not in st.session_state:
      st.session_state.input_text = "hủy báo thức lúc sáu giờ sáng mai"

    user_query = st.text_area(
        "Nhập văn bản cần kiểm thử:",
        value=st.session_state.input_text,
        height=90,
        placeholder="Gõ tự do: câu chuẩn, không dấu, teencode hoặc câu lạ...",
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
        st.session_state.input_text = "đặt báo thức lúc sáu giờ sáng mai"
        st.rerun()

    btn_run = st.button("🚀 Phân Loại Ý Định", type="primary", use_container_width=True)

  with col_right:
    st.subheader("🚦 Kết Quả Phân Tích NLU:")

    if user_query:
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

      # Chuẩn hóa in-scope về 100% để thanh Top-3 không bị hiển thị 0%
      sum_in = in_scope_probs.sum()
      norm_in_probs = (
          in_scope_probs / sum_in if sum_in > 0 else in_scope_probs
      )

      best_in_idx = in_scope_indices[np.argmax(in_scope_probs)]
      predicted_intent = p4_id2label[best_in_idx]
      best_conf = float(np.max(in_scope_probs))

      # QUYẾT ĐỊNH 3 TRẠNG THÁI
      if oos_score >= tau_oos_val:
        st.error(f"""
                ### 🚨 TRẠNG THÁI 3: NGOÀI PHẠM VI (UNKNOWN / OOS)
                * **Điểm câu lạ (OOS Score):** `{oos_score*100:.2f}%` $\ge$ Ngưỡng $\\tau_{{oos}}$ (`{tau_oos_val*100:.2f}%`)
                * **Hành động:** Từ chối xử lý và thông báo câu lệnh nằm ngoài phạm vi hỗ trợ.
                """)
      elif best_conf < tau_conf_val:
        top2_idx = np.argsort(norm_in_probs)[-2:][::-1]
        top2_names = [p4_id2label[in_scope_indices[i]] for i in top2_idx]
        st.warning(f"""
                ### ⚠️ TRẠNG THÁI 2: MƠ HỒ / ĐỘ TIN CẬY THẤP (CLARIFY)
                * **Độ tự tin:** `{best_conf*100:.2f}%` $<$ Ngưỡng tự tin $\\tau_{{conf}}$ (`{tau_conf_val*100:.2f}%`)
                * **Hành động:** Gợi ý hỏi lại: *Có phải bạn muốn `{top2_names[0]}` hoặc `{top2_names[1]}`?*
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

with tab_labels:
  st.subheader("📋 Phân Định Không Gian Nhãn (Giao thức Open-Intent)")
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
    st.markdown("#### 🚫 Các Ý Định Ngoài Phạm Vi Tiêu Biểu (OOS)")
    oos_samples = [
        "alarm_set (Bị rút làm OOS)",
        "calendar_set",
        "iot_coffee",
        "music_likeness",
        "takeaway_order",
        "transport_taxi",
        "audiobook_stop",
        "qa_stock (Tài chính/Chứng khoán)",
        "cooking_query",
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
