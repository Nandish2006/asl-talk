/**
 * ASLTalk v3
 * Fixes: sentence flicker, random light theme on load,
 *        full keyboard control, professional logo
 */

// ── SocketIO ──────────────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusDot       = document.getElementById('statusDot');
const statusText      = document.getElementById('statusText');
const modeTag         = document.getElementById('modeTag');
const videoHint       = document.getElementById('videoHint');
const detLetter       = document.getElementById('detLetter');
const detWord         = document.getElementById('detWord');
const detSuggested    = document.getElementById('detSuggested');
const detSentence     = document.getElementById('detSentence');
const btnBackspace    = document.getElementById('btnBackspace');
const btnSpace        = document.getElementById('btnSpace');
const btnApply        = document.getElementById('btnApply');
const btnClear        = document.getElementById('btnClear');
const btnSendSentence = document.getElementById('btnSendSentence');
const chatMessages    = document.getElementById('chatMessages');
const chatInput       = document.getElementById('chatInput');
const btnSend         = document.getElementById('btnSend');
const btnClearChat    = document.getElementById('btnClearChat');
const typingIndicator = document.getElementById('typingIndicator');
const predictionBox   = document.getElementById('predictionBox');
const predictionList  = document.getElementById('predictionList');
const kbHintEl        = document.getElementById('kbHint');
const kbBar           = document.getElementById('kbBar');

// ══════════════════════════════════════════════════════════════════════════════
// RANDOM LIGHT THEME — one per session, changes on refresh
// ══════════════════════════════════════════════════════════════════════════════
const THEMES = [
  { topbar:'#1e3a5f', accent:'#2980b9', dark:'#1a5276', chat:'#dce8f5', sent:'#c8e6fa', name:'Ocean Blue'   },
  { topbar:'#145a32', accent:'#1e8449', dark:'#0e6124', chat:'#d5f0e0', sent:'#b8e8c8', name:'Forest Green' },
  { topbar:'#4a235a', accent:'#7d3c98', dark:'#5b2c6f', chat:'#ead5f5', sent:'#d7b8f0', name:'Royal Purple' },
  { topbar:'#784212', accent:'#d35400', dark:'#a04000', chat:'#fde8d5', sent:'#faccaa', name:'Warm Amber'   },
  { topbar:'#1a5276', accent:'#2471a3', dark:'#154360', chat:'#d6eaf8', sent:'#aed6f1', name:'Steel Blue'   },
  { topbar:'#0b3d2e', accent:'#148f77', dark:'#0e6655', chat:'#d1f2eb', sent:'#a2d9ce', name:'Teal Green'   },
  { topbar:'#2c3e50', accent:'#566573', dark:'#1c2833', chat:'#eaecee', sent:'#d5d8dc', name:'Slate'        },
  { topbar:'#7b241c', accent:'#c0392b', dark:'#922b21', chat:'#fadbd8', sent:'#f1948a', name:'Crimson'      },
];

(function applyTheme() {
  // New random theme every page load (remove sessionStorage line to keep per-session)
  const t = THEMES[Math.floor(Math.random() * THEMES.length)];
  const s = document.documentElement.style;
  s.setProperty('--topbar-bg',   t.topbar);
  s.setProperty('--accent',      t.accent);
  s.setProperty('--accent-dark', t.dark);
  s.setProperty('--chat-bg',     t.chat);
  s.setProperty('--sent-bg',     t.sent);

  // Flash theme name
  const badge = document.getElementById('themeBadge');
  if (badge) {
    badge.textContent  = '🎨 ' + t.name;
    badge.style.opacity = '1';
    setTimeout(() => { badge.style.opacity = '0'; }, 3000);
  }
})();

// ══════════════════════════════════════════════════════════════════════════════
// FLICKER FIX — diff-based prediction rendering (never rebuilds unchanged chips)
// ══════════════════════════════════════════════════════════════════════════════
let _lastPreds = [];

function renderPredictions(predictions) {
  if (!predictionBox || !predictionList) return;

  // Skip re-render if predictions haven't changed
  const same = predictions.length === _lastPreds.length &&
    predictions.every((p, i) => p === _lastPreds[i]);
  if (same) return;
  _lastPreds = [...predictions];

  if (!predictions.length) {
    predictionBox.style.display = 'none';
    predictionList.innerHTML    = '';
    return;
  }

  predictionBox.style.display = 'block';
  const existing = Array.from(predictionList.querySelectorAll('.pred-chip'));

  predictions.forEach((text, i) => {
    if (existing[i]) {
      // Only update text if it changed — no DOM removal/re-add = no flicker
      existing[i].dataset.text = text;
      const span = existing[i].querySelector('.pred-text');
      if (span && span.textContent !== text) span.textContent = text;
    } else {
      const chip        = document.createElement('div');
      chip.className    = 'pred-chip';
      chip.dataset.text = text;
      chip.innerHTML    =
        `<span class="pred-num">${i + 1}</span>` +
        `<span class="pred-text">${text}</span>` +
        `<span class="pred-use">↵</span>`;
      chip.addEventListener('click', () => applyPrediction(chip.dataset.text));
      predictionList.appendChild(chip);
    }
  });

  // Remove extra chips if list shrank
  for (let i = predictions.length; i < existing.length; i++) existing[i].remove();
}

async function applyPrediction(text) {
  try {
    await fetch('/api/apply_prediction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  } catch (e) { console.error(e); }
}

// ══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function flashCard(el) {
  const card = el.closest('.det-card');
  if (!card) return;
  card.classList.remove('flash');
  void card.offsetWidth;
  card.classList.add('flash');
}

function updateField(el, value) {
  const display = value && value.trim() ? value : '—';
  if (el.textContent !== display) { el.textContent = display; flashCard(el); }
}

function scrollChat() { chatMessages.scrollTop = chatMessages.scrollHeight; }

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendMessage(sender, text, time) {
  const row    = document.createElement('div');
  row.className = `msg-row ${sender === 'user' ? 'sent' : 'received'}`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  const p  = document.createElement('p'); p.textContent = text;
  const ts = document.createElement('span');
  ts.className = 'msg-time'; ts.textContent = time || nowTime();
  bubble.appendChild(p); bubble.appendChild(ts);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollChat();
}

function showTyping(show) {
  typingIndicator.style.display = show ? 'flex' : 'none';
  if (show) scrollChat();
}

function pulse(el, color = '#25d366') {
  el.style.boxShadow = `0 0 0 4px ${color}55`;
  setTimeout(() => { el.style.boxShadow = ''; }, 350);
}

function flashBtn(el) {
  if (!el) return;
  el.style.transform = 'scale(0.91)';
  el.style.filter    = 'brightness(0.85)';
  setTimeout(() => { el.style.transform = ''; el.style.filter = ''; }, 160);
}

// ══════════════════════════════════════════════════════════════════════════════
// STATUS
// ══════════════════════════════════════════════════════════════════════════════
async function fetchStatus() {
  try {
    const d = await fetch('/api/status').then(r => r.json());
    if (d.demo_mode) {
      statusDot.className    = 'status-dot demo';
      statusText.textContent = 'Demo Mode';
      modeTag.textContent    = '⚠ DEMO';
      if (videoHint) videoHint.textContent = '⚠ No model — demo mode';
    } else {
      statusDot.className    = 'status-dot online';
      statusText.textContent = 'Model Loaded';
      modeTag.textContent    = '✓ LIVE';
      if (videoHint) videoHint.textContent = '📷 Detecting hand gestures…';
    }
  } catch {
    statusDot.className    = 'status-dot offline';
    statusText.textContent = 'Disconnected';
    modeTag.textContent    = '✕ OFFLINE';
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// SOCKET EVENTS
// ══════════════════════════════════════════════════════════════════════════════
socket.on('connect',    () => { statusDot.className = 'status-dot online'; statusText.textContent = 'Connected'; fetchStatus(); });
socket.on('disconnect', () => { statusDot.className = 'status-dot offline'; statusText.textContent = 'Disconnected'; });

socket.on('asl_update', (data) => {
  updateField(detLetter,    data.letter    || '');
  updateField(detWord,      data.word      || '');
  updateField(detSuggested, data.suggested || '');
  updateField(detSentence,  data.sentence  || '');
  renderPredictions(data.predictions || []);
});

socket.on('new_message', (msg) => {
  showTyping(false);
  // userA = ASL signer → right side (sent style)
  // userB = keyboard   → left side (received style)
  // Both arrive here for ALL connected clients
  const side = (msg.sender === 'userA') ? 'user' : 'bot';
  appendMessage(side, msg.text, msg.timestamp);
});

// ══════════════════════════════════════════════════════════════════════════════
// BUTTON CONTROLS
// ══════════════════════════════════════════════════════════════════════════════
async function postAction(endpoint) {
  try { return await fetch(endpoint, { method: 'POST' }).then(r => r.json()); }
  catch (e) { console.error(e); }
}

btnBackspace.addEventListener('click', () => postAction('/api/backspace'));
btnSpace.addEventListener('click',     () => postAction('/api/add_space'));
btnApply.addEventListener('click',     () => postAction('/api/apply_correction'));

btnClear.addEventListener('click', async () => {
  await postAction('/api/clear_sentence');
  updateField(detLetter, ''); updateField(detWord, '');
  updateField(detSuggested, ''); updateField(detSentence, '');
  renderPredictions([]);
});

btnSendSentence.addEventListener('click', async () => {
  const s = detSentence.textContent;
  if (!s || s === '—') { pulse(btnSendSentence, '#ef4444'); return; }
  await sendMessage(s, 'userA');
});

// ══════════════════════════════════════════════════════════════════════════════
// SEND MESSAGE — two-way: userA = ASL signer, userB = keyboard user
// Backend broadcasts via socket so BOTH sides see every message
// ══════════════════════════════════════════════════════════════════════════════
async function sendMessage(textOverride, sender) {
  const senderRole = sender || 'userB';
  const text = textOverride || chatInput.value.trim();
  if (!text && senderRole === 'userB') return;

  // Only clear the textarea for keyboard user
  if (senderRole === 'userB') {
    chatInput.value = '';
    autoResizeTextarea();
  }

  // Do NOT add bubble locally — backend emits new_message to ALL clients
  try {
    await fetch('/api/send_message', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, sender: senderRole }),
    });
  } catch (e) { appendMessage('bot', '❌ Failed. Retry.', nowTime()); }
}

btnSend.addEventListener('click', () => sendMessage('', 'userB'));
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage('', 'userB'); }
});

function autoResizeTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 90) + 'px';
}
chatInput.addEventListener('input', autoResizeTextarea);
btnClearChat.addEventListener('click', () => chatMessages.querySelectorAll('.msg-row').forEach(r => r.remove()));

// ══════════════════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS — full mouse-free control
// Key map: key → { hint, btn, action }
// ══════════════════════════════════════════════════════════════════════════════
const KB = [
  { key: ' ',         display: 'Space',     hint: 'commit word',       btn: btnSpace,        action: () => postAction('/api/add_space') },
  { key: 'Backspace', display: 'Backspace', hint: 'delete letter',     btn: btnBackspace,    action: () => postAction('/api/backspace') },
  { key: 'Enter',     display: 'Enter',     hint: 'send message',      btn: btnSendSentence, action: () => btnSendSentence.click() },
  { key: 'a',         display: 'A',         hint: 'apply word fix',    btn: btnApply,        action: () => postAction('/api/apply_correction') },
  { key: 'c',         display: 'C',         hint: 'clear sentence',    btn: btnClear,        action: () => btnClear.click() },
  { key: 'x',         display: 'X',         hint: 'clear / exit',      btn: btnClear,        action: () => btnClear.click() },
  { key: 'Escape',    display: 'Esc',       hint: 'reset all',         btn: btnClear,        action: () => btnClear.click() },
  { key: '1',         display: '1',         hint: 'prediction 1',      btn: null,            action: () => pickChip(0) },
  { key: '2',         display: '2',         hint: 'prediction 2',      btn: null,            action: () => pickChip(1) },
  { key: '3',         display: '3',         hint: 'prediction 3',      btn: null,            action: () => pickChip(2) },
];

function pickChip(idx) {
  const chips = predictionList ? predictionList.querySelectorAll('.pred-chip') : [];
  if (chips[idx]) chips[idx].click();
}

// Build keyboard shortcut bar
if (kbBar) {
  kbBar.innerHTML = KB.map(k =>
    `<span class="kb-key"><kbd>${k.display}</kbd><span>${k.hint}</span></span>`
  ).join('');
}

document.addEventListener('keydown', (e) => {
  if (document.activeElement === chatInput) return; // let user type freely

  const entry = KB.find(k => k.key === e.key);
  if (!entry) return;
  e.preventDefault();

  entry.action();
  flashBtn(entry.btn);

  // Show hint feedback
  if (kbHintEl) {
    kbHintEl.textContent   = `⌨  ${entry.display}  →  ${entry.hint}`;
    kbHintEl.style.opacity = '1';
    clearTimeout(kbHintEl._t);
    kbHintEl._t = setTimeout(() => { kbHintEl.style.opacity = '0'; }, 1600);
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════════════════
const videoFeed = document.getElementById('videoFeed');
if (videoFeed) videoFeed.addEventListener('error', () => { videoFeed.alt = '📷 Camera not available'; });

fetchStatus();
scrollChat();