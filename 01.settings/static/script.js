const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendButtonEl = document.getElementById("send-button");

// 지금까지의 대화 히스토리 (OpenAI messages 포맷)
let history = [];

function renderMessages() {
  messagesEl.innerHTML = "";
  for (const msg of history) {
    const div = document.createElement("div");
    div.className = `message ${msg.role}`;
    div.textContent = msg.content;
    messagesEl.appendChild(div);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showError(text) {
  const div = document.createElement("div");
  div.className = "message error";
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();

  const text = inputEl.value.trim();
  if (!text) return;

  history.push({ role: "user", content: text });
  renderMessages();
  inputEl.value = "";
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });

    if (!res.ok) {
      throw new Error(`서버 오류 (${res.status})`);
    }

    const data = await res.json();
    history = data.messages;
    renderMessages();
  } catch (err) {
    showError(`메시지 전송에 실패했습니다: ${err.message}`);
  } finally {
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  }
});
