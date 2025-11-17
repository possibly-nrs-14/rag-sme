let lastQa = null;
let lastQuiz = null;
let chatSessionId = "session_" + Date.now(); // Session ID for chat agent

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload || {})
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return res.json();
}

/**
 * Replaces window.alert() with a non-blocking toast message.
 */
function showToast(message, duration = 3000) {
  // Remove existing toast
  const existingToast = document.querySelector(".toast-message");
  if (existingToast) {
    existingToast.remove();
  }

  // Create toast element
  const toast = document.createElement("div");
  toast.className = "toast-message";
  toast.textContent = message;
  document.body.appendChild(toast);

  // Animate in
  setTimeout(() => {
    toast.classList.add("show");
  }, 10);

  // Animate out
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => {
      toast.remove();
    }, 500); // Wait for transition to end
  }, duration);
}


function renderSources(listEl, sources) {
  listEl.innerHTML = "";
  (sources || []).forEach(s => {
    const li = document.createElement("li");
    li.textContent =
      `${s.book} | chunk=${s.chunk_id} | g=${s.granularity} | pos=${s.position}`;
    listEl.appendChild(li);
  });
}

/**
 * Feedback functions removed.
 */

// --- NEW: Chat UI Functions ---

const chatMessages = document.getElementById("chat-messages");

function addChatMessage(message, type) {
    if (!chatMessages) return; // Guard clause
    
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${type}-message`;
    
    // Basic markdown replacement for AI messages
    if (type === 'ai' || type === 'ai-thinking') {
        
        let html_content = "";
        
        // --- FIX 4: Check if the message is an object (our quiz/QA JSON) ---
        if (typeof message === 'object' && message !== null && !Array.isArray(message)) {
            
            if (message.items) { // This is a Quiz object
                html_content = "Here is a quiz on your topic:<br><br>";
                message.items.forEach((it, idx) => {
                    // Sanitize text before adding
                    const question = (it.question || "").replace(/</g, "&lt;");
                    html_content += `<strong>${idx + 1}. ${question}</strong><br>`;
                    const opts = it.options || {};
                    ["A","B","C","D"].forEach(k => {
                        const option = (opts[k] || "").replace(/</g, "&lt;");
                        html_content += `&nbsp;&nbsp;&nbsp;${k}. ${option}<br>`;
                    });
                    const correct = (it.correct || "").replace(/</g, "&lt;");
                    html_content += `&nbsp;&nbsp;&nbsp;<em>(Correct: ${correct})</em><br><br>`;
                });
            } else if (message.answer) { // This is a QA object
                const answer = (message.answer || "").replace(/</g, "&lt;");
                html_content = answer.replace(/\n/g, '<br>'); // Render newlines
                html_content += "<br><br><strong>Sources:</strong><br>";
                (message.sources || []).forEach(s => {
                    const book = (s.book || "N/A").replace(/</g, "&lt;");
                    html_content += `• ${book} | chunk=${s.chunk_id}<br>`;
                });
            } else {
                // Fallback for other objects: pretty-print JSON
                html_content = `<pre>${JSON.stringify(message, null, 2)}</pre>`;
            }
        } else {
            // It's a regular string message
            let safeMessage = String(message).replace(/</g, "&lt;").replace(/>/g, "&gt;");
            html_content = safeMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
            html_content = html_content.replace(/\*(.*?)\*/g, '<em>$1</em>'); // Italic
            html_content = html_content.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>'); // Code blocks
            html_content = html_content.replace(/`(.*?)`/g, '<code>$1</code>'); // Inline code
            html_content = html_content.replace(/\n/g, '<br>'); // Newlines
        }
        
        msgDiv.innerHTML = html_content;
        
    } else {
        msgDiv.textContent = message; // User messages are plain text
    }
    
    chatMessages.appendChild(msgDiv);
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    const chatInput = document.getElementById("chat-input");
    const chatSend = document.getElementById("chat-send");
    
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable form
    chatInput.disabled = true;
    chatSend.disabled = true;
    chatInput.value = "";
    
    // Render user message
    addChatMessage(query, 'user');

    // Add thinking indicator
    addChatMessage("Thinking...", 'ai-thinking');

    try {
        // Send to /chat endpoint
        const res = await postJSON("/chat", {
            query: query,
            session_id: chatSessionId
        });
        
        // Remove thinking indicator
        const thinkingMsg = document.querySelector('.ai-thinking');
        if (thinkingMsg) {
            thinkingMsg.remove();
        }

        // Render AI response
        addChatMessage(res.response, 'ai');

    } catch (err) {
        // Remove thinking indicator
        const thinkingMsg = document.querySelector('.ai-thinking');
        if (thinkingMsg) {
            thinkingMsg.remove();
        }
        addChatMessage(`Error: ${err.message}`, 'ai');
        showToast(`Error: ${err.message}`);
    } finally {
        // Re-enable form
        chatInput.disabled = false;
        chatSend.disabled = false;
        chatInput.focus();
    }
}

// NEW: Function to handle clearing chat history
async function handleClearChat() {
    try {
        const res = await postJSON("/chat/clear", { session_id: chatSessionId });
        chatMessages.innerHTML = ""; // Clear the UI
        addChatMessage("Hello! How can I help you today?", 'ai'); // Add welcome message
        showToast(res.message || "Chat history cleared.");
    } catch (err) {
        showToast(`Error clearing history: ${err.message}`);
    }
}


// --- END: Chat UI Functions ---


window.addEventListener("DOMContentLoaded", () => {
  // --- Feedback UI creation removed ---
  
  const qaPanel = document.getElementById("qa-panel");
  const qaInput = document.getElementById("qa-input");
  const qaRun = document.getElementById("qa-run");
  const qaAnswer = document.getElementById("qa-answer");
  const qaSources = document.getElementById("qa-sources");
  const qaExport = document.getElementById("qa-export");
  // --- Feedback controls removed ---


  const quizPanel = document.getElementById("quiz-panel");
  const quizTopic = document.getElementById("quiz-topic");
  const quizCount = document.getElementById("quiz-count");
  const quizRun = document.getElementById("quiz-run");
  const quizPreview = document.getElementById("quiz-preview");
  const quizSources = document.getElementById("quiz-sources");
  const quizExport = document.getElementById("quiz-export");
  // --- Feedback controls removed ---
  
  // --- NEW: Chat Panel ---
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatClearBtn = document.getElementById("chat-clear"); // NEW

  // --- Feedback Modal Listeners Removed ---

  // --- QA Logic ---
  if (qaRun) {
      qaRun.addEventListener("click", async () => {
        const q = qaInput.value.trim();
        if (!q) return;
        qaRun.disabled = true;
        qaAnswer.textContent = "Thinking...";
        qaSources.innerHTML = "";
        // --- Feedback visibility line removed ---
        try {
          const res = await postJSON("/lc/qa", {question: q});
          lastQa = res; 
          qaAnswer.textContent = res.answer || "";
          renderSources(qaSources, res.sources || []);
          qaExport.disabled = !(res.answer && res.sources);
          // --- Feedback visibility line removed ---
        } catch (err) {
          qaAnswer.textContent = `Error: ${err.message}`;
          qaExport.disabled = true;
        } finally {
          qaRun.disabled = false;
        }
      });
  }

  if (qaExport) {
      qaExport.addEventListener("click", async () => {
        if (!lastQa) return;
        qaExport.disabled = true;
        try {
          const res = await postJSON("/export/qa", {
            answer: lastQa.answer || "",
            sources: lastQa.sources || []
          });
          showToast("Exported: " + (res.paths || []).join(", "));
        } catch (err) {
          showToast("Export failed: " + err.message);
        } finally {
          qaExport.disabled = false;
        }
      });
  }

  // --- Quiz Logic ---
  if (quizRun) {
      quizRun.addEventListener("click", async () => {
        const topic = quizTopic.value.trim();
        const n = parseInt(quizCount.value || "5", 10);
        if (!topic) return;
        quizRun.disabled = true;
        quizPreview.textContent = "Generating quiz...";
        if (quizSources) quizSources.innerHTML = "";                 
        try {
          const res = await postJSON("/lc/quiz", {topic, n});
          lastQuiz = res;
          const items = (res.items || []);
          let text = "";
          items.forEach((it, idx) => {
            text += `${idx + 1}. ${it.question}\n`;
            const opts = (it.options || {});
            ["A","B","C","D"].forEach(k => {
              text += `   ${k}. ${opts[k] || ""}\n`;
            });
            text += `   Correct: ${it.correct}\n\n`;
          });
          quizPreview.textContent = text || "No items generated.";
          if (quizSources) {
            renderSources(quizSources, res.sources || []);
          }
          quizExport.disabled = items.length === 0;
          // --- Feedback visibility line removed ---
        } catch (err) {
          quizPreview.textContent = `Error: ${err.message}`;
          quizExport.disabled = true;
        } finally {
          quizRun.disabled = false;
        }
      });
  }
  
  if (quizExport) {
      quizExport.addEventListener("click", async () => {
        if (!lastQuiz) return;
        quizExport.disabled = true;
        try {
          const res = await postJSON("/export/quiz", lastQuiz);
          showToast("Exported: " + (res.paths || []).join(", "));
        } catch (err) {
          showToast("Export failed: " + err.message);
        } finally {
          quizExport.disabled = false;
        }
      });
  }

  // --- Global Feedback Button Listeners Removed ---

  // --- NEW: Chat Form Listener ---
  if (chatForm) {
      chatForm.addEventListener("submit", handleChatSubmit);
  }
  
  // Allow Enter to send, Shift+Enter for newline in chat
  if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
          if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleChatSubmit(e);
          }
      });
  }
  
  // NEW: Clear Button Listener
  if (chatClearBtn) {
      chatClearBtn.addEventListener("click", handleClearChat);
  }

});