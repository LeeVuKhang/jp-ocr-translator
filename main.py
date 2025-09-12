import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from PIL import Image
import pytesseract
from fugashi import Tagger

# Load biến môi trường
load_dotenv()

# Đường dẫn Tesseract (tùy máy bạn)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Init MeCab
tagger = Tagger()

# Init Groq LLM
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# Streamlit UI
st.set_page_config(page_title="AI học tiếng Nhật", page_icon="🇯🇵", layout="centered")
st.title("🇯🇵 AI OCR + Dịch + Chú thích học tiếng Nhật")

uploaded_file = st.file_uploader("Tải lên ảnh tiếng Nhật (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # OCR với Tesseract
    image = Image.open(uploaded_file)
    japanese_text = pytesseract.image_to_string(image, lang="jpn", config="--psm 6")

    st.subheader("📄 Văn bản OCR được")
    st.text_area("Nội dung", japanese_text, height=200)

    # Nút dịch + phân tích
    if st.button("➡️ Dịch & Chú thích (gộp chung)"):
        with st.spinner("⏳ Đang xử lý..."):
            # Phân tích MeCab trước
            words = []
            for word in tagger(japanese_text):
                surface = word.surface
                reading = word.feature.kana if hasattr(word.feature, "kana") else ""
                pos = word.feature.pos if hasattr(word.feature, "pos") else ""
                words.append(f"{surface} ({reading}) - {pos}")

            mecab_analysis = "\n".join(words)

            # Gửi sang LLM: cả đoạn + phân tích
            response = model.invoke([
                HumanMessage(
                    content=f"""
Hãy dịch đoạn văn tiếng Nhật này sang tiếng Việt và đồng thời làm bảng chú thích học tiếng Nhật (Kanji/Hiragana/Ý nghĩa).
Văn bản OCR:
{japanese_text}

Phân tích từ vựng (MeCab):
{mecab_analysis}
""",
                    additional_kwargs={"job_role": "Japanese language teacher"}
                )
            ])

        st.subheader("📖 Bản dịch + Chú thích (gộp)")
        st.write(response.content)
        st.success("✅ Hoàn tất!")


def clean_mecab(japanese_text):
    words = []
    buffer = ""

    for word in tagger(japanese_text):
        surface = word.surface
        reading = word.feature.kana if hasattr(word.feature, "kana") else ""
        pos = word.feature.pos if hasattr(word.feature, "pos") else ""

        # Loại bỏ ký tự vô nghĩa như dấu chấm câu, ký tự rời lẻ
        if pos in ["記号", "空白"]:
            if buffer:
                words.append(buffer)
                buffer = ""
            continue

        # Gom katakana liên tiếp
        if surface.iskatakana():
            buffer += surface
        else:
            if buffer:
                words.append(buffer)
                buffer = ""
            words.append(surface)

    if buffer:
        words.append(buffer)

    return words