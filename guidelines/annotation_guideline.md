# 📋 Hướng Dẫn Gán Nhãn Dữ Liệu (Annotation Guidelines) - Đề tài 2

## 1. Tổng quan
Tài liệu này quy định chuẩn gán nhãn ngữ nghĩa (Intent Classification) và phát hiện ngoài phạm vi (Out-of-Scope) cho bài toán trợ lý ảo tiếng Việt trên tập dữ liệu MASSIVE và PhoATIS.

## 2. Quy tắc gán nhãn In-Scope
- **Miền (Scenario):** Mỗi câu lệnh thuộc một kịch bản cụ thể (ví dụ: `alarm`, `calendar`, `iot`, `music`,...).
- **Ý định (Intent):** Phải phản ánh đúng hành động mong muốn của người dùng (ví dụ: `alarm_set`, `alarm_remove`).

## 3. Quy tắc gán nhãn Out-of-Scope (OOS)
- Các câu lệnh không thuộc 40 danh mục In-Scope được gán nhãn cố định là `out_of_scope`.
- Dùng để kiểm tra khả năng từ chối câu lệnh lạ của mô hình (OOS Detection).
