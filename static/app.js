let lastQa = null;
let lastQuiz = null;
let chatSessionId = "session_" + Date.now(); // Session ID for chat agent

// ============================================================================
// LLM Configuration Management
// ============================================================================

const LLM_CONFIG_KEY = "sme_llm_config";

function loadLLMConfig() {
    const saved = localStorage.getItem(LLM_CONFIG_KEY);
    if (saved) {
        try {
            return JSON.parse(saved);
        } catch (e) {
            return null;
        }
    }
    return null;
}

function saveLLMConfig(config) {
    localStorage.setItem(LLM_CONFIG_KEY, JSON.stringify(config));
}

function updateConfigUI(config) {
    const modelSelect = document.getElementById("config-model");
    const tempInput = document.getElementById("config-temperature");
    const tokensInput = document.getElementById("config-max-tokens");
    const statusDiv = document.getElementById("config-status");

    if (modelSelect && config.model_name) {
        modelSelect.value = config.model_name;
    }
    if (tempInput && config.temperature !== undefined) {
        tempInput.value = config.temperature;
    }
    if (tokensInput && config.max_new_tokens !== undefined) {
        tokensInput.value = config.max_new_tokens;
    }
    if (statusDiv) {
        statusDiv.textContent = `Active: ${config.model_name || 'MediPhi'}`;
        statusDiv.style.color = "#10b981";
    }
}

async function applyLLMConfig() {
    const modelSelect = document.getElementById("config-model");
    const tempInput = document.getElementById("config-temperature");
    const tokensInput = document.getElementById("config-max-tokens");
    const statusDiv = document.getElementById("config-status");
    const applyBtn = document.getElementById("config-apply");

    if (!modelSelect || !tempInput || !tokensInput) return;

    // Get current saved config to detect if model is changing
    const oldConfig = loadLLMConfig();
    const modelChanging = oldConfig && oldConfig.model_name !== modelSelect.value;

    const config = {
        model_name: modelSelect.value,
        temperature: parseFloat(tempInput.value),
        max_new_tokens: parseInt(tokensInput.value, 10)
    };

    if (applyBtn) applyBtn.disabled = true;
    if (statusDiv) statusDiv.textContent = "Applying configuration...";

    try {
        const res = await postJSON("/admin/llm-config", config);
        saveLLMConfig(res.config);
        updateConfigUI(res.config);

        if (modelChanging) {
            showToast(`Model changed to ${res.config.model_name}. New model will load on next query (may take 10-30 seconds).`);
            if (statusDiv) {
                statusDiv.textContent = `Active: ${res.config.model_name} (will load on next request)`;
                statusDiv.style.color = "#f59e0b"; // Orange to indicate pending
            }
        } else {
            showToast("LLM configuration updated successfully!");
            if (statusDiv) {
                statusDiv.textContent = `Active: ${res.config.model_name}`;
                statusDiv.style.color = "#10b981"; // Green
            }
        }
    } catch (err) {
        if (statusDiv) {
            statusDiv.textContent = `Error: ${err.message}`;
            statusDiv.style.color = "#ef4444";
        }
        showToast(`Config update failed: ${err.message}`);
    } finally {
        if (applyBtn) applyBtn.disabled = false;
    }
}

async function initLLMConfig() {
    // Load saved config or fetch current from server
    let config = loadLLMConfig();

    if (!config) {
        try {
            const res = await fetch("/admin/llm-config");
            if (res.ok) {
                config = await res.json();
                saveLLMConfig(config);
            }
        } catch (e) {
            console.warn("Could not fetch LLM config:", e);
        }
    }

    if (config) {
        updateConfigUI(config);
    }
}

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

// ============================================================================
// Intermediate Steps Rendering
// ============================================================================

function renderIntermediateSteps(steps) {
    if (!steps || steps.length === 0) return null;

    const container = document.createElement("div");
    container.className = "intermediate-steps";

    const summary = document.createElement("summary");
    summary.textContent = "Show reasoning";
    summary.className = "steps-summary";

    const details = document.createElement("details");
    details.appendChild(summary);

    const stepsContent = document.createElement("div");
    stepsContent.className = "steps-content";

    steps.forEach((step, index) => {
        const stepDiv = document.createElement("div");
        stepDiv.className = "step-item";

        // Always show step number for clarity
        const stepHeader = document.createElement("div");
        stepHeader.style.fontWeight = "600";
        stepHeader.style.color = "#60a5fa";
        stepHeader.style.marginBottom = "0.5rem";
        stepHeader.textContent = `Step ${index + 1}`;
        stepDiv.appendChild(stepHeader);

        if (step.thought && step.thought.trim()) {
            const thoughtP = document.createElement("p");
            thoughtP.innerHTML = `<strong>Thought:</strong> ${escapeHtml(step.thought)}`;
            stepDiv.appendChild(thoughtP);
        }

        if (step.action) {
            const actionP = document.createElement("p");
            const actionInput = typeof step.action_input === 'object'
                ? JSON.stringify(step.action_input, null, 2)
                : String(step.action_input || '');
            actionP.innerHTML = `<strong>Action:</strong> ${escapeHtml(step.action)}`;
            if (actionInput && actionInput !== '{}' && actionInput !== '') {
                actionP.innerHTML += `<br><strong>Input:</strong> <code>${escapeHtml(actionInput)}</code>`;
            }
            stepDiv.appendChild(actionP);
        }

        if (step.observation) {
            const obsP = document.createElement("p");
            const obsText = String(step.observation).substring(0, 300);
            obsP.innerHTML = `<strong>Observation:</strong> ${escapeHtml(obsText)}${step.observation.length > 300 ? '...' : ''}`;
            stepDiv.appendChild(obsP);
        }

        stepsContent.appendChild(stepDiv);
    });

    details.appendChild(stepsContent);
    container.appendChild(details);

    return container;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
    const statusDiv = document.getElementById("config-status");

    const query = chatInput.value.trim();
    if (!query) return;

    // Disable form
    chatInput.disabled = true;
    chatSend.disabled = true;
    chatInput.value = "";

    // Render user message
    addChatMessage(query, 'user');

    // Add thinking indicator (enhanced for model loading)
    const savedConfig = loadLLMConfig();
    const modelName = savedConfig ? savedConfig.model_name.split('/').pop() : 'model';
    addChatMessage(`Thinking (loading ${modelName} if needed)...`, 'ai-thinking');

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

        // Update status to show model is now loaded
        if (statusDiv && statusDiv.textContent.includes('will load on next request')) {
            const currentConfig = loadLLMConfig();
            if (currentConfig) {
                statusDiv.textContent = `Active: ${currentConfig.model_name}`;
                statusDiv.style.color = "#10b981"; // Green - now actually loaded
            }
        }

        // Render AI response
        addChatMessage(res.response, 'ai');

        // Render intermediate steps if available
        if (res.intermediate_steps && res.intermediate_steps.length > 0) {
            const stepsElement = renderIntermediateSteps(res.intermediate_steps);
            if (stepsElement && chatMessages) {
                chatMessages.appendChild(stepsElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }

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
  // Initialize LLM configuration
  initLLMConfig();

  // Add Apply button listener for config
  const configApplyBtn = document.getElementById("config-apply");
  if (configApplyBtn) {
      configApplyBtn.addEventListener("click", applyLLMConfig);
  }

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
          lastQa = { question: q, ...res };
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
            question: lastQa.question || "",
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
          lastQuiz = { topic, ...res };
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