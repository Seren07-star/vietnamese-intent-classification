import os
import torch
import torch.nn.functional as F
import gradio as gr
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

MODEL_NAME = "vinai/phobert-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Loading model on {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    model.to(device)
    model.eval()
    has_model = True
except Exception as e:
    print(f"⚠️ Fallback mode: {e}")
    has_model = False

def predict_intent(text, tau_threshold, top_k=5):
    if not text or text.strip() == "":
        return "⚠️ Vui lòng nhập câu lệnh để dự đoán.", {}, ""
    
    segmented_text = ViTokenizer.tokenize(text.strip())
    
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
    top_k_results = {k: v for k, v in sorted_probs[:top_k]}
    
    if max_p < tau_threshold or top_intent == "out_of_scope":
        status_msg = f"🚨 **CẢNH BÁO: NGOÀI MIỀN (OOS)**\n- Độ tin cậy tối đa: `{max_p:.2%}` < Ngưỡng $\\tau$ (`{tau_threshold:.2%}`)."
        decision_display = "⚠️ [UNKNOWN / OOS]"
    else:
        status_msg = f"✅ **HỢP LỆ (IN-SCOPE)**\n- Ý định: **{top_intent}**\n- Độ tin cậy: `{max_p:.2%}`"
        decision_display = f"🎯 {top_intent} ({max_p:.2%})"
        
    return decision_display, top_k_results, status_msg

demo = gr.Blocks(title="Vietnamese Intent Classification")
with demo:
    gr.Markdown("# 🎙️ Vietnamese Intent Classification & OOS Detection")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Nhập câu lệnh:", placeholder="Ví dụ: đặt báo thức 6h sáng mai...")
            tau_slider = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Ngưỡng tin cậy (Threshold τ)")
            btn_submit = gr.Button("Phân tích ý định", variant="primary")
        with gr.Column():
            out_decision = gr.Label(label="Quyết định hệ thống")
            out_probs = gr.Label(label="Top-k Xác suất")
            out_status = gr.Markdown(label="Trạng thái")

    btn_submit.click(predict_intent, inputs=[input_text, tau_slider], outputs=[out_decision, out_probs, out_status])

if __name__ == "__main__":
    demo.launch()
