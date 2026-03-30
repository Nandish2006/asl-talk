/**
 * ASLTalk — Frontend Logic
 * ─────────────────────────────────────────────────────
 * · Real two-way chat (no bot replies)
 * · Suggestion chips from dictionary_word.txt only
 * · User CHOOSES to apply suggestion or keep own word
 * · Sender switcher (ASL User ↔ Other Person)
 */

// ── Socket ────────────────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

// ── DOM refs ──────────────────────────────────────────────────────────────────
const connDot          = document.getElementById('connDot');
const connLabel        = document.getElementById('connLabel');
const modeBadge        = document.getElementById('modeBadge');

const detLetter        = document.getElementById('detLetter');
const detWord          = document.getElementById('detWord');
const detSentence      = document.getElementById('detSentence');
const suggChips        = document.getElementById('suggChips');

const btnBackspace     = document.getElementById('btnBackspace');
const btnSpace         = document.getElementById('btnSpace');
const btnClear         = document.getElementById('btnClear');
const btnClearAll      = document.getElementById('btnClearAll');
const btnSendSentence  = document.getElementById('btnSendSentence');
const btnKeepWord      = document.getElementById('btnKeepWord');

const chatMessages     = document.getElementById('chatMessages');
const chatInput        = document.getElementById('chatInput');
const btnSend          = document.getElementById('btnSend');
const btnClearChat     = document.getElementById('btnClearChat');
const senderTabs       = document.querySelectorAll('.sender-tab');

// ── State ─────────────────────────────────────────────────────────────────────
let activeSender    = 'user-asl';   // 'user-asl' or 'user-b'
let currentWord     = '';
let currentSugg     = [];           // current suggestion list
let wordAccepted    = false;        // user chose a suggestion
let lastSentWord    = '';           // for de-duplication

// ── Helpers ───────────────────────────────────────────────────────────────────
const nowTime = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

function scrollChat() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function flashTile(el) {
  const tile = el.closest('.det-tile');
  if (!tile) return;
  tile.classList.remove('flash');
  void tile.offsetWidth;
  tile.classList.add('flash');
}

function setField(el, val) {
  const v = (val && val.trim()) ? val : '—';
  if (el.textContent !== v) {
    el.textContent = v;
    flashTile(el);
  }
}

// ── Connection status ─────────────────────────────────────────────────────────
socket.on('connect', () => {
  connDot.className   = 'conn-dot online';
  connLabel.textContent = 'Connected';
  fetchStatus();
});
socket.on('disconnect', () => {
  connDot.className   = 'conn-dot offline';
  connLabel.textContent = 'Disconnected';
});

async function fetchStatus() {
  try {
    const d = await (await fetch('/api/status')).json();
    if (d.demo_mode) {
      modeBadge.textContent = 'Demo';
      modeBadge.className   = 'mode-badge demo';
    } else {
      modeBadge.textContent = 'Live';
      modeBadge.className   = 'mode-badge live';
    }
  } catch { /* ignore */ }
}

// ── ASL real-time update ──────────────────────────────────────────────────────
socket.on('asl_update', (data) => {
  setField(detLetter,   data.letter   || '');
  setField(detWord,     data.word     || '');
  setField(detSentence, data.sentence || '');

  const word = (data.word || '').trim();

  // Fetch suggestions from server whenever word changes
  if (word && word !== currentWord) {
    currentWord  = word;
    wordAccepted = false;
    fetchSuggestions(word);
  } else if (!word) {
    currentWord = '';
    renderSuggestions([]);
  }
});

// ── Suggestions ───────────────────────────────────────────────────────────────
async function fetchSuggestions(word) {
  if (!word || word.length < 2) {
    renderSuggestions([]);
    return;
  }
  try {
    const r = await fetch(`/api/suggestions?word=${encodeURIComponent(word)}`);
    const d = await r.json();
    currentSugg = d.suggestions || [];
    renderSuggestions(currentSugg);
  } catch {
    renderSuggestions([]);
  }
}

function renderSuggestions(words) {
  suggChips.innerHTML = '';

  if (!words || words.length === 0) {
    const em = document.createElement('span');
    em.className   = 'sugg-empty';
    em.textContent = currentWord
      ? 'No close match found in dictionary'
      : 'Start typing a word…';
    suggChips.appendChild(em);
    return;
  }

  words.forEach((w, i) => {
    const chip = document.createElement('button');
    chip.className   = 'sugg-chip' + (i === 0 ? ' best-chip' : '');
    chip.textContent = w;
    chip.title       = i === 0 ? 'Best match from dictionary' : 'Alternative suggestion';
    chip.addEventListener('click', () => applySuggestion(w));
    suggChips.appendChild(chip);
  });
}

async function applySuggestion(word) {
  wordAccepted = true;
  // Tell backend to set current_word to the chosen word
  await post('/api/apply_correction', { word });
}

// Keep word as-is — user explicitly ignores suggestions
btnKeepWord.addEventListener('click', () => {
  wordAccepted = true;
  renderSuggestions([]);   // clear chips to show user's choice is locked in
  const em = document.createElement('span');
  em.className   = 'sugg-empty';
  em.textContent = `Keeping: ${currentWord}`;
  suggChips.appendChild(em);
});

// ── Incoming chat messages (broadcasted by server) ────────────────────────────
socket.on('new_message', (msg) => {
  // If this broadcast is the echo of our own send, skip it — we already
  // rendered it optimistically in sendMessage() below.
  if (msg.skip_sid && msg.skip_sid === socket.id) return;
  appendMessage(msg.sender, msg.text, msg.timestamp);
});

// ── Append message bubble ─────────────────────────────────────────────────────
function appendMessage(sender, text, time) {
  const isASL = sender === 'user-asl';
  const isB   = sender === 'user-b';

  const row    = document.createElement('div');
  row.className = `msg-row ${isASL ? 'msg-sent' : isB ? 'msg-b' : 'msg-recv'}`;

  const wrap   = document.createElement('div');
  wrap.style.maxWidth = '100%';

  // Sender label
  const label  = document.createElement('div');
  label.className   = 'msg-sender-label';
  label.textContent = isASL ? '🤟 You (ASL)' : isB ? '💬 Other Person' : '🤖 System';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  const p  = document.createElement('p');
  p.textContent = text;

  const ts = document.createElement('span');
  ts.className   = 'msg-ts';
  ts.textContent = time || nowTime();

  bubble.appendChild(p);
  bubble.appendChild(ts);
  wrap.appendChild(label);
  wrap.appendChild(bubble);
  row.appendChild(wrap);
  chatMessages.appendChild(row);
  scrollChat();
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(textOverride) {
  const text = (textOverride || chatInput.value).trim();
  if (!text) return;

  chatInput.value = '';
  autoResize();

  // Optimistic render — show immediately without waiting for server
  appendMessage(activeSender, text, nowTime());

  try {
    await post('/api/send_message', {
      text,
      sender: activeSender,
      sid: socket.id,       // ← server echoes this back as skip_sid
    });
  } catch {
    appendMessage('recv', '❌ Failed to send.', nowTime());
  }
}

btnSend.addEventListener('click', () => sendMessage());
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Send ASL sentence ─────────────────────────────────────────────────────────
btnSendSentence.addEventListener('click', async () => {
  const sentence = detSentence.textContent;
  if (!sentence || sentence === '—') return;
  // Force ASL sender when sending from the detection panel
  const prev = activeSender;
  activeSender = 'user-asl';
  await sendMessage(sentence);
  activeSender = prev;
});

// ── Sender tab switcher ───────────────────────────────────────────────────────
senderTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    senderTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeSender = tab.dataset.sender;
    chatInput.placeholder = activeSender === 'user-asl'
      ? 'Type as ASL User…'
      : 'Type as Other Person…';
  });
});

// ── Control buttons ───────────────────────────────────────────────────────────
btnBackspace.addEventListener('click', () => post('/api/backspace'));
btnSpace.addEventListener('click',     () => post('/api/add_space'));

btnClear.addEventListener('click', async () => {
  await post('/api/clear_sentence');
  currentWord = '';
  wordAccepted = false;
  renderSuggestions([]);
  setField(detLetter,   '');
  setField(detWord,     '');
  setField(detSentence, '');
});

btnClearAll.addEventListener('click', () => btnClear.click());

btnClearChat.addEventListener('click', () => {
  chatMessages.querySelectorAll('.msg-row').forEach(r => r.remove());
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (document.activeElement === chatInput) return;
  if (e.key === 'Backspace')  { e.preventDefault(); post('/api/backspace'); }
  if (e.key === ' ')          { e.preventDefault(); post('/api/add_space'); }
  if (e.key === 'Enter')      { e.preventDefault(); btnSendSentence.click(); }
});

// ── Textarea auto-resize ──────────────────────────────────────────────────────
function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 90) + 'px';
}
chatInput.addEventListener('input', autoResize);

// ── Generic POST helper ───────────────────────────────────────────────────────
async function post(url, body) {
  const opts = { method: 'POST' };
  if (body) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body    = JSON.stringify(body);
  }
  try {
    const r = await fetch(url, opts);
    return await r.json();
  } catch (e) {
    console.error(url, e);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
fetchStatus();
scrollChat();