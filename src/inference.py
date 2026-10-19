import torch
import torch.nn.functional as F
from pyvi import ViTokenizer

def predict_utterance(text, model, tokenizer, device, label2id, id2label, tau=0.7):
    """Hàm suy luận (inference) cho 1 câu đầu vào"""
    model.eval()
    segmented = ViTokenizer.tokenize(text.strip())
    inputs = tokenizer(segmented, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    max_prob = float(max(probs))
    pred_idx = int(probs.argmax())
    pred_label = id2label[pred_idx]
    
    if max_prob < tau or pred_label == "out_of_scope":
        return {"decision": "OUT_OF_SCOPE", "confidence": max_prob, "intent": "out_of_scope"}
    return {"decision": "IN_SCOPE", "confidence": max_prob, "intent": pred_label}
