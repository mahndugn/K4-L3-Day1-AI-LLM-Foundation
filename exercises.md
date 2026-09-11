# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, mô hình phản hồi mang tính quy chuẩn, các lần gọi lặp lại cho kết quả y hệt nhau. Khi tăng lên 0.5 và 1.0, các sự thật được chia sẻ đa dạng và tự nhiên hơn (như văn hóa cà phê, hang Sơn Đoòng, bờ biển), từ ngữ sinh động hơn. Khi đạt mức 1.5, câu văn bắt đầu phóng đại quá mức, cấu trúc ngữ pháp lỏng lẻo và dễ xuất hiện hiện tượng ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng 0.0 đến 0.3 cho chatbot hỗ trợ khách hàng. Lý do là vì nghiệp vụ CSKH đòi hỏi tính chính xác tuyệt đối, sự nhất quán về chính sách, giá cả và điều khoản bảo hành; temperature thấp giúp hạn chế tối đa việc mô hình bịa đặt hoặc đưa ra thông tin mâu thuẫn giữa các lần hỏi.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng output token mỗi ngày là 10.000 * 3 * 350 = 10.500.000 tokens. Chi phí GPT-4o ($0.010/1k) là $105/ngày, trong khi GPT-4o-mini ($0.0006/1k) chỉ tốn $6.3/ngày; do đó GPT-4o đắt hơn khoảng 16.7 lần (~17 lần). Trường hợp GPT-4o xứng đáng chi phí: Phân tích báo cáo tài chính chuyên sâu, soát xét hợp đồng pháp lý hoặc sinh code logic phức tạp. Trường hợp nên dùng mini: Phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn hoặc trả lời các câu hỏi thường gặp (FAQ) chuẩn mực.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi phân hóa rất rõ ràng: Bản giáo viên tiểu học ngắn gọn, từ ngữ thân thiện và dùng hình ảnh ẩn dụ cuốn sổ ghi chép chung của cả lớp mà không ai tẩy xóa được. Bản chuyên gia tài chính dài hơn, phân tích chuyên sâu qua các thuật ngữ kỹ thuật như sổ cái phân tán (distributed ledger), thuật toán đồng thuận (consensus mechanism), mã hóa mật mã học và tính phi tập trung. System prompt hoạt động như một bộ chỉ dẫn bối cảnh (context steering), định hình phong cách, trường từ vựng và cấp độ tư duy của mô hình cho phù hợp với đối tượng mục tiêu mà không cần sửa câu hỏi user.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, ước lượng tiếng Anh (100 / 0.75) cho ra khoảng 133 token, nhưng tiktoken đo thực tế dao động từ 180 đến 210 token (chênh lệch từ 35% đến 58%). Tiếng Việt tốn nhiều token hơn vì bộ tokenizer (BPE) của OpenAI được huấn luyện áp đảo trên dữ liệu tiếng Anh; các từ tiếng Việt có dấu thanh và nguyên âm ghép (như ơ, ư, ă, ê, đ...) thường không nằm trọn trong từ điển mà bị băm nhỏ thành nhiều sub-words hoặc byte tokens riêng rẽ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác thời gian thực (web chat, app di động, CLI) với các câu trả lời dài; nó giúp giảm thời gian chờ token đầu tiên (TTFT) xuống dưới 1 giây, mang lại cảm giác mượt mà và người dùng có thể đọc ngay khi AI đang suy nghĩ. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (background jobs), xử lý theo lô (batch processing), chấm điểm tự động, hoặc khi hệ thống yêu cầu phản hồi dạng JSON có cấu trúc (Structured Outputs) cần được parse toàn vẹn trước khi chuyển tiếp cho module khác.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có ưu thế tự động giãn cách thời gian chờ tăng dần theo lũy thừa (0.1s -> 0.2s -> 0.4s -> 0.8s...), giúp giảm mật độ request và tạo điều kiện cho máy chủ giải tỏa hàng nghẽn quá tải. Nếu hàng nghìn client cùng retry với khoảng delay cố định 1 giây, tất cả client sẽ đồng loạt dồn request tấn công vào server tại cùng một mốc thời gian, tạo nên hiện tượng 'Thundering Herd' (bão request chu kỳ) làm server tiếp tục bị nghẽn và không thể hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: 'Bạn là trợ giảng thân thiện của khóa học AI LLM, giải thích các khái niệm kỹ thuật rõ ràng bằng ví dụ thực tế, luôn trả lời ngắn gọn và bằng tiếng Việt.' Lựa chọn 'trả lời ngắn gọn' giúp kiểm soát độ dài câu trả lời, tiết kiệm token chi phí và tránh làm học viên bị ngợp thông tin trong môi trường dòng lệnh CLI. Lựa chọn 'bằng tiếng Việt' đảm bảo trợ lý duy trì ngôn ngữ mẹ đẻ ổn định, không tự ý chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật chuyên sâu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là lịch sử chỉ lưu tối đa 3 lượt gần nhất và mất toàn bộ ngữ cảnh khi kết thúc phiên, khiến trợ lý không nhớ được thông tin người dùng đã cung cấp từ trước. Đề xuất cải thiện: Triển khai cơ chế Tóm tắt hội thoại (Conversation Summarization) kết hợp Bộ nhớ ngoài (SQLite / Vector DB). Cụ thể: sau mỗi 3 lượt, gọi một model nhỏ (như gpt-4o-mini) tạo bản tóm tắt các điểm then chốt của cuộc trò chuyện và chèn nội dung tóm tắt đó vào một system message bổ sung, giúp AI giữ được bối cảnh lâu dài mà không làm phình to chi phí token input.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
