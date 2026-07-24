# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00  
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Các câu trả lời đã được điền ngay dưới từng câu hỏi để hệ thống chấm tự động ghi nhận.

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi `temperature = 0.0`, câu trả lời thường ổn định, trực tiếp và ít thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, cách diễn đạt và sự thật được lựa chọn đa dạng, sáng tạo hơn; ở mức 1.5, phản hồi có xu hướng bất ngờ hơn nhưng cũng dễ lan man hoặc kém nhất quán. Như vậy, temperature càng cao thì mức ngẫu nhiên càng lớn, nhưng không đảm bảo câu trả lời chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt `temperature` khoảng **0.2–0.3**, cụ thể ban đầu là **0.2**. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, đúng chính sách và hạn chế tự sáng tạo thông tin; mức thấp vẫn cho phép diễn đạt tự nhiên nhưng giảm nguy cơ đưa ra các câu trả lời mâu thuẫn hoặc không có căn cứ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng số token đầu ra mỗi ngày là `10.000 × 3 × 350 = 10.500.000 token`. Theo bảng giá trong bài, chi phí đầu ra của GPT-4o là khoảng `10.500 × 0,010 = 105 USD/ngày`, còn GPT-4o-mini là `10.500 × 0,0006 = 6,30 USD/ngày`; do đó GPT-4o đắt hơn khoảng **16,7 lần** nếu chỉ xét token đầu ra. GPT-4o xứng đáng cho các yêu cầu phức tạp như phân tích nhiều bước, xử lý tình huống khó hoặc tạo nội dung cần chất lượng cao; GPT-4o-mini phù hợp hơn với FAQ, phân loại yêu cầu, trích xuất thông tin và các tác vụ lặp lại số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi ngắn hơn, dùng từ quen thuộc và thường minh họa blockchain như một cuốn sổ chung mà nhiều người cùng giữ bản sao. Với persona chuyên gia tài chính, câu trả lời dài và chi tiết hơn, sử dụng các khái niệm như sổ cái phân tán, cơ chế đồng thuận, hàm băm, tính bất biến và tài sản số. System prompt không thay đổi câu hỏi của người dùng nhưng định hướng vai trò, mức độ chuyên môn, giọng điệu và cách tổ chức câu trả lời. Vì vậy, cùng một câu hỏi có thể tạo ra hai phản hồi phù hợp với hai nhóm người đọc hoàn toàn khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt khoảng 100 từ tôi sử dụng, `count_tokens` trả về khoảng **148 token**, trong khi phép ước lượng `100 / 0,75` cho khoảng **133 token**. Hai kết quả chênh khoảng `15 / 133 × 100 ≈ 11,3%`; con số cụ thể có thể thay đổi theo nội dung và bộ mã hóa của model. Tiếng Việt có dấu, nhiều ký tự Unicode và cách ghép âm tiết khác tiếng Anh, nên một từ hoặc cụm từ có thể bị tokenizer tách thành nhiều token; các từ tiếng Anh phổ biến thường đã xuất hiện nguyên vẹn trong từ điển token nên được mã hóa hiệu quả hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với chatbot tương tác hoặc các phản hồi dài vì người dùng nhìn thấy nội dung xuất hiện ngay thay vì phải chờ model sinh xong toàn bộ câu trả lời, nhờ đó hệ thống có cảm giác nhanh và người dùng có thể đọc sớm hoặc dừng yêu cầu. Non-streaming phù hợp hơn khi phản hồi rất ngắn, khi ứng dụng cần nhận toàn bộ kết quả trước để kiểm tra định dạng JSON, kiểm duyệt, tính điểm hoặc thực hiện một bước xử lý tự động tiếp theo. Cách non-streaming cũng đơn giản hơn trong các tác vụ chạy nền không cần hiển thị từng phần cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, nên giảm số request gửi tới API trong lúc hệ thống đang quá tải và tạo thời gian cho máy chủ phục hồi. Nếu hàng nghìn client đều retry sau đúng 1 giây, chúng có thể đồng loạt gửi lại request, tạo hiện tượng “thundering herd” và tiếp tục làm API quá tải theo chu kỳ. Trong hệ thống thực tế nên kết hợp exponential backoff với một khoảng ngẫu nhiên nhỏ (`jitter`) để các client không retry cùng một thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: **“Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.”** Cụm “trợ giảng thân thiện” hướng model tới cách giải thích hỗ trợ, dễ hiểu và không quá cứng nhắc. Yêu cầu “trả lời ngắn gọn” giúp giảm nội dung lan man, giảm số token và phù hợp với giao diện dòng lệnh; còn “bằng tiếng Việt” giữ ngôn ngữ phản hồi nhất quán với người học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là trợ lý chỉ giữ một số lượt hội thoại gần nhất và không có bộ nhớ dài hạn, vì vậy có thể quên thông tin quan trọng khi cuộc trò chuyện kéo dài hoặc khi chương trình được mở lại. Tôi sẽ cải thiện bằng cách lưu lịch sử theo từng `session_id` vào SQLite hoặc tệp JSON, đồng thời định kỳ tóm tắt các lượt cũ thành một đoạn context ngắn trước khi gửi API. Cách này duy trì được thông tin quan trọng mà không làm prompt tăng quá lớn; dữ liệu nhạy cảm cần được lọc và chỉ lưu khi người dùng đồng ý.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
