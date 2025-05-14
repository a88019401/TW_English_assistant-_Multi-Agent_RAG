// static/js/script.js
async function sendMessage() {
  const input = document.getElementById("input");
  const userInput = input.value.trim();
  if (!userInput) return;

  const chatBox = document.getElementById("chatBox");
  chatBox.innerHTML += `<div class="user">🙋‍♂️ ${escapeHtml(userInput)}</div>`;
  input.value = "";

  // 顯示思考中
  const thinkingId = `thinking-${Date.now()}`;
  chatBox.innerHTML += `<div class="bot" id="${thinkingId}">🤖 AI 正在思考中...</div>`;

  try {
    const res = await fetch("/ask_multiagent_rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: userInput }),
    });

    const data = await res.json();
    const thinkingDiv = document.getElementById(thinkingId);

    if (data.error) {
      thinkingDiv.innerHTML = `❌ 錯誤：${escapeHtml(data.error)}`;
    } else if (data.assistant_answer && data.practice_questions) {
      thinkingDiv.innerHTML = `
        <b>🎓 助教解答：</b><br>${escapeHtml(data.assistant_answer).replaceAll(
          "\n",
          "<br>"
        )}
        <br><br><b>📋 題目練習：</b><br>${escapeHtml(
          data.practice_questions
        ).replaceAll("\n", "<br>")}
      `;
    } else {
      thinkingDiv.innerHTML = `❓ 無回應`;
    }
  } catch (error) {
    console.error(error);
    const thinkingDiv = document.getElementById(thinkingId);
    thinkingDiv.innerHTML = `❌ 發生錯誤：${escapeHtml(error.message)}`;
  }
}

// 避免 XSS 攻擊
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
