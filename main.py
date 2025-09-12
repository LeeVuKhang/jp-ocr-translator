import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq   
from dotenv import load_dotenv
import os
import io
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract

load_dotenv() 

st.set_page_config(page_title="AI dịch tiếng Nhật", page_icon="🤖", layout = "centered")

st.title("🤖 AI dịch tiếng Nhật")
st.markdown("Ứng dụng dịch tiếng Nhật sử dụng mô hình ngôn ngữ lớn (LLM) và các công cụ tích hợp.")

model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0
    )

uploaded_file = st.file_uploader(
    "Tải lên file văn bản/ảnh tiếng Nhật (txt, docx, pdf, png, jpg, jpeg)", 
    type=["txt", "docx", "pdf", "png", "jpg", "jpeg"]
)

def read_file(uploaded_file):
    text = ""
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image, lang="jpn")  # OCR tiếng Nhật
    return text

if uploaded_file is not None:
    japanese_text = read_file(uploaded_file)
    st.subheader("📄 Văn bản gốc (tiếng Nhật OCR/Text)")
    st.text_area("Nội dung", japanese_text, height=200)

    if st.button("Dịch sang Tiếng Việt"):
        response = model.invoke([
            HumanMessage(
                content=f"Dịch đoạn văn bản tiếng Nhật sau sang tiếng Việt, kèm chú thích học tiếng Nhật (Kanji - Hiragana - Nghĩa):\n\n{japanese_text}",
                additional_kwargs={"job_role": "Japanese language teacher"}  
            )
        ])
        st.subheader("📖 Bản dịch & Chú thích")
        st.write(response.content)
