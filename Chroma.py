# Chroma.py
import os
import glob
import json
from openai import OpenAI    # ✅ 改成新版
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# 初始化
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # ✅ 新版初始化

chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))

def extract_text_from_pdf(pdf_path):
    """讀取 PDF 並回傳文字"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"⚠️ 頁面讀取錯誤：{e}")
    except Exception as e:
        print(f"❌ 無法讀取 PDF：{pdf_path}，錯誤：{e}")
    return text

def split_text(text, chunk_size=500):
    """把文字切成小段"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text):
    """呼叫 OpenAI API 產生向量"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

def build_manuals_collection():
    """建立 PDF手冊向量資料庫"""
    manuals_collection = chroma_client.get_or_create_collection(name="manuals")

    if not os.path.exists("data/user_manuals"):
        print("⚠️ 找不到資料夾 data/user_manuals，跳過 PDF 建立")
        return

    pdf_files = glob.glob("data/user_manuals/*.pdf")
    print(f"✅ 共找到 {len(pdf_files)} 本PDF教材")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"📄 正在處理：{filename}")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"⚠️ 檔案無內容，略過：{filename}")
            continue

        chunks = split_text(text)

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            manuals_collection.add(
                ids=[f"{filename}_{idx}"],
                documents=[chunk],
                embeddings=[embedding]
            )
    print("✅ PDF手冊資料已建立完畢")

def build_questions_collection():
    """建立 題庫向量資料庫"""
    questions_collection = chroma_client.get_or_create_collection(name="english_questions")
    json_path = "data/english_question_bank.json"

    if not os.path.exists(json_path):
        print("⚠️ 找不到題庫 json：data/english_question_bank.json，跳過建立")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ 共載入 {len(data)} 題題庫資料")

    for item in data:
        text = f"[{item['type']}] {item['question']} Answer: {item['answer']}"
        embedding = get_embedding(text)

        questions_collection.add(
            ids=[item['id']],
            documents=[text],
            embeddings=[embedding]
        )
    print("✅ 題庫資料已建立完畢")

def main():
    print("🎯 開始建立 Chroma 向量資料庫")
    build_manuals_collection()
    build_questions_collection()
    print("🎉 Chroma資料庫已完成！可以開始使用")

if __name__ == "__main__":
    main()
