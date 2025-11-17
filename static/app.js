let lastQa = null;
let lastQuiz = null;

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

function renderSources(listEl, sources) {
  listEl.innerHTML = "";
  (sources || []).forEach(s => {
    const li = document.createElement("li");
    li.textContent =
      `${s.book} | chunk=${s.chunk_id} | g=${s.granularity} | pos=${s.position}`;
    listEl.appendChild(li);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  const qaInput = document.getElementById("qa-input");
  const qaRun = document.getElementById("qa-run");
  const qaAnswer = document.getElementById("qa-answer");
  const qaSources = document.getElementById("qa-sources");
  const qaExport = document.getElementById("qa-export");

  const quizTopic = document.getElementById("quiz-topic");
  const quizCount = document.getElementById("quiz-count");
  const quizRun = document.getElementById("quiz-run");
  const quizPreview = document.getElementById("quiz-preview");
    const quizSources = document.getElementById("quiz-sources");
  const quizExport = document.getElementById("quiz-export");

  qaRun.addEventListener("click", async () => {
    const q = qaInput.value.trim();
    if (!q) return;
    qaRun.disabled = true;
    qaAnswer.textContent = "Thinking...";
    qaSources.innerHTML = "";
    try {
      const res = await postJSON("/lc/qa", {question: q});
      lastQa = res;
      qaAnswer.textContent = res.answer || "";
      renderSources(qaSources, res.sources || []);
      qaExport.disabled = !(res.answer && res.sources);
    } catch (err) {
      qaAnswer.textContent = `Error: ${err.message}`;
      qaExport.disabled = true;
    } finally {
      qaRun.disabled = false;
    }
  });

  qaExport.addEventListener("click", async () => {
    if (!lastQa) return;
    qaExport.disabled = true;
    try {
      const res = await postJSON("/export/qa", {
        answer: lastQa.answer || "",
        sources: lastQa.sources || []
      });
      alert("Exported: " + (res.paths || []).join(", "));
    } catch (err) {
      alert("Export failed: " + err.message);
    } finally {
      qaExport.disabled = false;
    }
  });

  quizRun.addEventListener("click", async () => {
    const topic = quizTopic.value.trim();
    const n = parseInt(quizCount.value || "5", 10);
    if (!topic) return;
    quizRun.disabled = true;
    quizPreview.textContent = "Generating quiz...";
    if (quizSources) quizSources.innerHTML = "";                  // clear old sources
    try {
      const res = await postJSON("/lc/quiz", {topic, n});
      lastQuiz = res;
      const items = (res.items || []);
      let text = "";
      items.forEach((it, idx) => {
        text += `${idx + 1}. ${it.question}\n`;
        const opts = it.options || {};
        ["A","B","C","D"].forEach(k => {
          text += `   ${k}. ${opts[k] || ""}\n`;
        });
        text += `   Correct: ${it.correct}\n\n`;
      });
      quizPreview.textContent = text || "No items generated.";

      // NEW: render quiz sources
      if (quizSources) {
        renderSources(quizSources, res.sources || []);
      }
      
      quizExport.disabled = items.length > 0;
      quizExport.disabled = items.length === 0;
    } catch (err) {
      quizPreview.textContent = `Error: ${err.message}`;
      quizExport.disabled = true;
    } finally {
      quizRun.disabled = false;
    }
  });

  quizExport.addEventListener("click", async () => {
    if (!lastQuiz) return;
    quizExport.disabled = true;
    try {
      const res = await postJSON("/export/quiz", lastQuiz);
      alert("Exported: " + (res.paths || []).join(", "));
    } catch (err) {
      alert("Export failed: " + err.message);
    } finally {
      quizExport.disabled = false;
    }
  });
});
