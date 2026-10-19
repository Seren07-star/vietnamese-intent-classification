import os
import re
import json
import random
import unicodedata
import pandas as pd
from pyvi import ViTokenizer

# 1. Danh sách Teencode & Viết tắt chuẩn giao tiếp tiếng Việt
TEENCODE_PAIRS = [
    ("như thế nào", "ntn"), ("làm sao", "sao"), ("thế nào", "tnao"),
    ("hôm nay", "hni"), ("ngày mai", "ngmai"), ("bây giờ", "h"),
    ("thông tin", "ttin"), ("cảm ơn", "tks"), ("xin chào", "hi"),
    ("không thể", "ko the"), ("không", "ko"), ("chẳng", "chag"),
    ("được", "dc"), ("người", "ng"), ("gì", "j"), ("mình", "mk"),
    ("rồi", "r"), ("quá", "qá"), ("biết", "bít"), ("phải", "fai"),
    ("bao nhiêu", "bn"), ("chút", "chut"), ("đang", "dag"), ("muốn", "mún")
]

class VietnameseAugmenter:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        
    def remove_diacritics(self, text):
        """Chuyển văn bản tiếng Việt thành không dấu chuẩn 100% ASCII"""
        text_nfd = unicodedata.normalize('NFD', text)
        text_no_accent = re.sub(r'[\u0300-\u036f]', '', text_nfd)
        text_clean = unicodedata.normalize('NFC', text_no_accent)
        return text_clean.replace('đ', 'd').replace('Đ', 'D')
        
    def replace_teencode(self, text):
        """Thay thế các từ và cụm từ thông dụng bằng Teencode"""
        res = text.lower()
        modified = False
        for phrase, teen in TEENCODE_PAIRS:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, res):
                res = re.sub(pattern, teen, res)
                modified = True
        if not modified:
            res = res.replace("ch", "c").replace("ng", "g").replace("kh", "k")
        return res
        
    def add_keyboard_typo(self, text):
        """Mô phỏng lỗi gõ trượt phím và gõ Telex thực tế"""
        words = text.split()
        if not words:
            return text
        idx = self.rng.randint(0, len(words) - 1)
        w = words[idx]
        telex_map = {"đ": "dd", "ê": "ee", "ơ": "ow", "ư": "uw", "â": "aa", "ô": "oo"}
        for k, v in telex_map.items():
            if k in w:
                words[idx] = w.replace(k, v)
                return " ".join(words)
        if len(w) >= 3:
            p = self.rng.randint(0, len(w) - 2)
            w = w[:p] + w[p+1] + w[p] + w[p+2:]
        words[idx] = w
        return " ".join(words)
        
    def augment(self, text):
        dice = self.rng.random()
        if dice < 0.40:
            return self.remove_diacritics(text), "no_diacritic"
        elif dice < 0.75:
            return self.replace_teencode(text), "teencode"
        else:
            return self.add_keyboard_typo(text), "typo"
