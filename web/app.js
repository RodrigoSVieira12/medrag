"use strict";

const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const remainingEl = document.getElementById("remaining");
const statusEl = document.getElementById("status");
const sourcesEl = document.getElementById("sources");
const sourcesList = document.getElementById("sources-list");
const answerWrap = document.getElementById("answer-wrap");
const answerEl = document.getElementById("answer");

function show(el) {
  el.classList.remove("hidden");
}
function hide(el) {
  el.classList.add("hidden");
}

function setStatus(text, state) {
  statusEl.className = "status" + (state ? " " + state : "");
  statusEl.textContent = text;
  show(statusEl);
}

function updateRemaining(n) {
  if (n === null || n === undefined) return;
  remainingEl.textContent =
    n > 0 ? `${n} question${n === 1 ? "" : "s"} left` : "Demo limit reached";
}

// Turn inline [n] markers into links that jump to the matching source.
function linkifyCitations(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped.replace(
    /\[(\d+)\]/g,
    (_, n) => `<a class="cite" href="#source-${n}">[${n}]</a>`
  );
}

function renderSources(sources) {
  sourcesList.innerHTML = "";
  for (const s of sources) {
    const li = document.createElement("li");
    li.id = `source-${s.n}`;
    const links = [`<a href="${s.url}" target="_blank" rel="noopener">PubMed</a>`];
    if (s.oa_url) {
      links.push(
        `<a href="${s.oa_url}" target="_blank" rel="noopener">Open access</a>`
      );
    }
    li.innerHTML =
      `<div class="src-title">${s.title}</div>` +
      `<div class="src-cite">${s.citation}</div>` +
      `<div>${links.join("")}</div>`;
    sourcesList.appendChild(li);
  }
  show(sourcesEl);
}

async function ask(question) {
  hide(sourcesEl);
  hide(answerWrap);
  answerEl.textContent = "";
  setStatus("Searching PubMed and reading papers…");
  askBtn.disabled = true;

  let answerText = "";

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    // Rate-limit / validation errors come back as a single JSON object.
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      updateRemaining(data.remaining);
      setStatus(data.error || `Request failed (${resp.status}).`, "error");
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete newline-delimited JSON events.
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const evt = JSON.parse(line);

        if (evt.type === "sources") {
          renderSources(evt.sources);
          updateRemaining(evt.remaining);
          setStatus("Writing answer…");
          show(answerWrap);
        } else if (evt.type === "delta") {
          answerText += evt.text;
          answerEl.textContent = answerText;
        } else if (evt.type === "error") {
          setStatus(evt.message, "error");
          return;
        } else if (evt.type === "done") {
          answerEl.innerHTML = linkifyCitations(answerText);
          setStatus("Done", "done");
          hide(statusEl);
          return;
        }
      }
    }
  } catch (err) {
    setStatus("Connection problem. Please try again.", "error");
  } finally {
    askBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (q) ask(q);
});

// Show the session's remaining-question count on load.
fetch("/api/status")
  .then((r) => r.json())
  .then((d) => updateRemaining(d.remaining))
  .catch(() => {});
