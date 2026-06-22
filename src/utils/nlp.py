import numpy as np
from underthesea import word_tokenize  # Sử dụng thư viện tách từ tiếng Việt


def tokenize(sentence: str):
    """
    Tách câu tiếng Việt thành mảng các từ/cụm từ.
    Ví dụ: "Học máy rất thú vị" -> ['Học máy', 'rất', 'thú vị']
    """
    return word_tokenize(sentence.lower())


def stem(word: str):
    """
    Chuẩn hóa từ về chữ thường và xóa khoảng trắng thừa.
    """
    return word.lower().strip()


def bag_of_words(tokenized_sentence, all_words):
    """
    Trả về mảng biểu diễn nhị phân (0 hoặc 1) sự xuất hiện của từ trong từ điển.
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0

    return bag