# rag_utils.py
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("請設定 .env 裡的 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_manual_collection():
    return chroma_client.get_or_create_collection(name="manuals")

def get_question_collection():
    return chroma_client.get_or_create_collection(name="english_questions")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

def search_manual_chunks(query_text, top_k=3):
    embedding = get_embedding(query_text)
    results = get_manual_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    documents = results["documents"][0] if results["documents"] else []
    return "\n---\n".join(documents)

def search_question_bank(query_text, top_k=5):
    embedding = get_embedding(query_text)
    results = get_question_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    documents = results["documents"][0] if results["documents"] else []
    return "\n---\n".join(documents)

def run_agent1_retrieve_context(user_prompt):
    manual = search_manual_chunks(user_prompt)
    questions = search_question_bank(user_prompt)
    return f"【教材內容】\n{manual}\n\n【題庫內容】\n{questions}"

def run_agent2_generate_examples(user_prompt, agent1_answer, question_context):
    followup_prompt = f"""學生原始問題：
{user_prompt}

以下是教材助理提供的解答內容：
{agent1_answer}

以下是題庫相關內容：
{question_context}

請根據以上內容，設計5題英文練習題（每題需含4個選項與答案），請直接開始："""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是資深的英語題目設計專家，請協助設計練習題。"},
            {"role": "user", "content": followup_prompt}
        ]
    )
    return response.choices[0].message.content
