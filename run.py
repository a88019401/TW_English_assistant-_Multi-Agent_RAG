# ✅ V3.2 強化版 Multi-Agent + Chroma RAG + Session 記憶 + Agent1+3合併

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from openai import OpenAI
from rag_utils import search_manual_chunks, search_question_bank
from dotenv import load_dotenv
import os
import re

app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_answer_pattern(text):
    return bool(re.fullmatch(r"[abcdABCD]{3,10}", text.strip()))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask_multiagent_rag", methods=["POST"])
def ask_multiagent_rag():
    data = request.json
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "請輸入問題"}), 400

    print(f"[User] {user_prompt}")

    manual_context = search_manual_chunks(user_prompt)
    question_context = search_question_bank(user_prompt)

    if not manual_context:
        manual_context = "⚠️ 找不到教材資料"
    if not question_context:
        question_context = "⚠️ 找不到題庫資料"

    # ✅ 若為答案模式
    if ("答案" in user_prompt or is_answer_pattern(user_prompt)) and session.get("last_practice_questions"):
        last_questions = session["last_practice_questions"]
        agent_answer_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是英文助教兼解題老師。請根據【教材內容】【題庫內容】【5題英文題目】提供每題正確答案與詳細解析。格式：1. 答案：（選項字母）+解析。"},
                {"role": "user", "content": f"學生問題：{session.get('last_topic', '')}\n\n【教材內容】\n{manual_context}\n\n【題庫內容】\n{question_context}\n\n5題英文題目：\n{last_questions}\n學生答案：{user_prompt}"}
            ]
        )
        agent_answer = agent_answer_response.choices[0].message.content
        print(f"[Agent1+3 Answer] {agent_answer}...")
        session["current_mode"] = None
        return jsonify({
            "question": user_prompt,
            "assistant_answer": agent_answer,
            "practice_questions": "(上一題練習題，未重複提供)"
        })

    # ✅ 正常流程
    last_questions = session.get("last_practice_questions", "（目前尚無歷史題目）")

    agent1_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是英文助教，僅參考【教材內容】【題庫內容】【歷史題目】詳細解釋學生問題。內容需簡潔、符合國中程度。"},
            {"role": "user", "content": f"學生問題：{user_prompt}\n\n【教材內容】\n{manual_context}\n\n【題庫內容】\n{question_context}\n\n【歷史題目】\n{last_questions}"}
        ]
    )
    agent1_answer = agent1_response.choices[0].message.content
    print(f"[Agent1 Explanation] {agent1_answer}...")

    agent2_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是英文測驗題目設計老師。根據【教材內容】【題庫內容】【助教回答】設計5題英文單選題（a/b/c/d），不得超出主題。只提供題目，不提供答案。"},
            {"role": "user", "content": f"學生問題：{user_prompt}\n\n助教回答：{agent1_answer}\n\n題庫內容：{question_context}"}
        ]
    )
    agent2_answer = agent2_response.choices[0].message.content
    print(f"[Agent2 Questions] {agent2_answer}...")

    session["last_practice_questions"] = agent2_answer
    session["last_topic"] = user_prompt
    session["current_mode"] = "waiting_for_answer"

    return jsonify({
        "question": user_prompt,
        "assistant_answer": agent1_answer,
        "practice_questions": agent2_answer
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)