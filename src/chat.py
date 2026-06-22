import os

# Import các module nội bộ của dự án
from src.data.persona_chat import load_persona_chat_pairs
from src.retrieval import find_best_response, retrieve_top_k_responses
from src.utils.nlp import bag_of_words, tokenize

# ──────────────────────────────────────────────
# 1. Tải dataset persona-chat
# ──────────────────────────────────────────────
print("Đang tải persona-chat dataset...")
conversations = load_persona_chat_pairs(split="train", max_examples=5000)
if not conversations:
    print("[-] Không tìm thấy dữ liệu persona-chat")
    exit()

print(f"[+] Đã tải {len(conversations)} cặp hội thoại")

# ──────────────────────────────────────────────
# 2. Xây dựng từ điển từ vựng từ tất cả input
# ──────────────────────────────────────────────
all_words = set()
for conv in conversations:
    tokens = tokenize(conv["input"])
    all_words.update(tokens)

all_words = sorted(list(all_words))
print(f"[+] Kích thước từ điển: {len(all_words)}")

# ──────────────────────────────────────────────
# 3. Vòng lặp trò chuyện (Retrieval-based)
# ──────────────────────────────────────────────
bot_name = "FPBot"
print("==================================================")
print(f"{bot_name}: Xin chào! Tôi đã sẵn sàng trò chuyện. (Gõ 'quit' để thoát)")
print("==================================================")

while True:
    sentence = input("Bạn: ")
    if sentence.lower() == "quit":
        break

    response, confidence = find_best_response(
        sentence, conversations, all_words, top_k=1
    )

    # Chỉ trả lời nếu confidence đủ cao
    if confidence > 0.15:
        print(f"{bot_name}: {response}")
    else:
        print(f"{bot_name}: Tôi chưa học đủ để trả lời chính xác câu hỏi này.")
