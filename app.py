"""
ASLTalk — Flask + SocketIO Backend
· Dictionary correction ONLY from dictionary_word.txt
· Two-way real chat (no auto-bot replies)
· Suggestion is optional — user chooses to apply or ignore
"""

import os, cv2, numpy as np, time, threading, random
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from collections import deque
from difflib import get_close_matches

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed – DEMO mode")

try:
    from tensorflow.keras.models import load_model as tf_load
    TF_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model as tf_load
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("[WARN] TensorFlow not installed – DEMO mode")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asltalk_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH  = 'asl_landmark_model.h5'
LABELS_PATH = 'labels.npy'
MEAN_PATH   = 'mean.npy'
STD_PATH    = 'std.npy'

# ── Global ML state ───────────────────────────────────────────────────────────
model      = None
labels     = None
mean_vals  = None
std_vals   = None
mp_hands   = None
hands      = None
mp_drawing = None

# ── ASL detection state ───────────────────────────────────────────────────────
current_letter   = ''
current_word     = ''
current_sentence = ''
suggested_word   = ''

SMOOTHING_WINDOW  = 10
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
last_letter       = ''
last_letter_time  = 0
BASE_COOLDOWN     = 1.2
REPEAT_COOLDOWN   = 3.3

# ── Chat state ────────────────────────────────────────────────────────────────
chat_messages  = []          # list of {sender, text, timestamp}
camera_active  = False
cap            = None
camera_lock    = threading.Lock()

# ── Demo mode ─────────────────────────────────────────────────────────────────
DEMO_LETTERS = list("HELLO WORLD")
demo_index   = 0
demo_timer   = 0

# ── Dictionary — ONLY from dictionary_word.txt ────────────────────────────────
WORDS = []   # ordered list — preserves priority
WORDS_SET = set()

_dict_path = 'dictionary_word.txt'
if os.path.exists(_dict_path):
    with open(_dict_path, encoding='utf-8') as f:
        for line in f:
            w = line.strip().lower()
            if w.isalpha() and w not in WORDS_SET:
                WORDS.append(w)
                WORDS_SET.add(w)
    print(f"[OK] Dictionary loaded from {_dict_path} — {len(WORDS)} words")
else:
    # tiny fallback so app never crashes
    WORDS = ['hello','help','good','yes','no','please','thanks','love',
             'want','need','come','go','make','take','give','call']
    WORDS_SET = set(WORDS)
    print("[WARN] dictionary_word.txt not found — using minimal fallback")


def get_suggestions(word, n=3):
    """
    Return up to n closest matches from dictionary_word.txt ONLY.
    Uses difflib with a reasonable cutoff so only genuinely close words appear.
    """
    if not word or len(word) < 2:
        return []
    w = word.lower()
    # difflib against our curated list
    matches = get_close_matches(w, WORDS, n=n, cutoff=0.55)
    return [m.upper() for m in matches]


def best_suggestion(word):
    """Return single best suggestion or empty string."""
    s = get_suggestions(word, n=1)
    return s[0] if s else ''


# ── Model loading ─────────────────────────────────────────────────────────────
def load_asl_model():
    global model, labels, mean_vals, std_vals, mp_hands, hands, mp_drawing

    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = tf_load(MODEL_PATH)
            print(f"[OK] Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERR] Model load failed: {e}")

    if os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"[OK] Labels: {labels}")

    if os.path.exists(MEAN_PATH):
        mean_vals = np.load(MEAN_PATH)
    if os.path.exists(STD_PATH):
        std_vals  = np.load(STD_PATH)

    if MEDIAPIPE_AVAILABLE:
        mp_hands   = mp.solutions.hands
        hands      = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        print("[OK] MediaPipe Hands ready")


# ── Prediction helpers ────────────────────────────────────────────────────────
def extract_landmarks(frame_rgb):
    if not MEDIAPIPE_AVAILABLE or hands is None:
        return None, None
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for pt in lm.landmark:
        coords.extend([pt.x, pt.y])
    return np.array(coords, dtype=np.float32), lm


def predict_letter(landmarks):
    if model is None or labels is None:
        return None
    try:
        if std_vals is not None and mean_vals is not None:
            norm = (landmarks - mean_vals) / (float(std_vals) + 1e-8) \
                   if std_vals.ndim == 0 \
                   else (landmarks - mean_vals) / (std_vals + 1e-8)
        else:
            norm = landmarks
        inp  = norm.reshape(1, -1)
        pred = model.predict(inp, verbose=0)
        idx  = np.argmax(pred)
        conf = float(pred[0][idx])
        lbl  = str(labels[idx])
        thr  = 0.85 if lbl in ('A','S','E','T') else \
               0.90 if lbl in ('R','U','V') else 0.70
        return lbl if conf >= thr else None
    except Exception as e:
        print(f"[ERR] predict: {e}")
        return None


def get_smoothed_letter():
    if not prediction_buffer:
        return None
    from collections import Counter
    counts = Counter(l for l in prediction_buffer if l is not None)
    if not counts:
        return None
    best, freq = counts.most_common(1)[0]
    return best if freq / SMOOTHING_WINDOW > 0.6 else None


def demo_letter():
    global demo_index, demo_timer
    now = time.time()
    if now - demo_timer > 1.5:
        demo_timer = now
        demo_index = (demo_index + 1) % len(DEMO_LETTERS)
    return DEMO_LETTERS[demo_index]


# ── Camera frame generator ────────────────────────────────────────────────────
def gen_frames():
    global cap, camera_active, current_letter, current_word
    global current_sentence, suggested_word, last_letter, last_letter_time

    with camera_lock:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_active = True

    try:
        while camera_active:
            ok, frame = cap.read()
            if not ok:
                break

            frame   = cv2.flip(frame, 1)
            display = frame.copy()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detected_letter = None

            if MEDIAPIPE_AVAILABLE and model is not None:
                landmarks, lm_obj = extract_landmarks(rgb)
                if landmarks is not None:
                    if mp_drawing and lm_obj:
                        mp_drawing.draw_landmarks(
                            display, lm_obj, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(34,197,94),  thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(16,185,129), thickness=2)
                        )
                    prediction_buffer.append(predict_letter(landmarks))
                    detected_letter = get_smoothed_letter()
                else:
                    prediction_buffer.append(None)
            else:
                detected_letter = demo_letter()

            now = time.time()
            if detected_letter and detected_letter != ' ':
                same   = (detected_letter == last_letter)
                gap    = now - last_letter_time
                allow  = (gap > REPEAT_COOLDOWN) if same else (gap > BASE_COOLDOWN)
                if allow:
                    current_letter   = detected_letter
                    last_letter      = detected_letter
                    last_letter_time = now
                    if detected_letter not in ('del', 'space', 'nothing'):
                        current_word  += detected_letter
                        suggested_word = best_suggestion(current_word) \
                                         if len(current_word) > 2 else ''

            # HUD overlay
            h, w = display.shape[:2]
            ov = display.copy()
            cv2.rectangle(ov, (0, h-110), (w, h), (15, 23, 42), -1)
            cv2.addWeighted(ov, 0.6, display, 0.4, 0, display)

            def put(txt, y, clr=(255,255,255), sz=0.58):
                cv2.putText(display, txt, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, sz, (0,0,0), 3)
                cv2.putText(display, txt, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, sz, clr, 1)

            put(f"Letter  : {current_letter}",  h-88, (100,255,150))
            put(f"Word    : {current_word}",     h-64, (255,255,255))
            put(f"Suggest : {suggested_word}",   h-40, (255,210,60))
            put(f"Sentence: {current_sentence}", h-16, (160,200,255))

            socketio.emit('asl_update', {
                'letter':    current_letter,
                'word':      current_word,
                'suggested': suggested_word,
                'sentence':  current_sentence,
            })

            ok2, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok2:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
    finally:
        with camera_lock:
            if cap:
                cap.release()
            camera_active = False


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded':    model is not None,
        'mediapipe_ready': MEDIAPIPE_AVAILABLE,
        'demo_mode':       model is None or not MEDIAPIPE_AVAILABLE,
    })


@app.route('/api/suggestions')
def suggestions_api():
    """Return top-3 suggestions for the current word from dictionary_word.txt"""
    word = request.args.get('word', current_word)
    return jsonify({'suggestions': get_suggestions(word, n=3)})


@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_letter, current_word, current_sentence, suggested_word
    current_letter = current_word = current_sentence = suggested_word = ''
    prediction_buffer.clear()
    socketio.emit('asl_update',
                  {'letter':'','word':'','suggested':'','sentence':''})
    return jsonify({'status': 'ok'})


@app.route('/api/backspace', methods=['POST'])
def backspace():
    global current_word, suggested_word, current_sentence
    if current_word:
        current_word   = current_word[:-1]
        suggested_word = best_suggestion(current_word) if len(current_word) > 2 else ''
    elif current_sentence:
        parts = current_sentence.strip().split()
        if parts:
            current_word     = parts[-1]
            current_sentence = ' '.join(parts[:-1])
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': current_word,
        'suggested': suggested_word, 'sentence': current_sentence
    })
    return jsonify({'word': current_word, 'sentence': current_sentence})


@app.route('/api/apply_correction', methods=['POST'])
def apply_correction():
    """User explicitly chose to apply the suggestion."""
    global current_word, suggested_word
    data = request.get_json(silent=True) or {}
    chosen = data.get('word', suggested_word)   # frontend can pass chosen word
    if chosen:
        current_word   = chosen
        suggested_word = ''
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': current_word,
        'suggested': suggested_word, 'sentence': current_sentence
    })
    return jsonify({'word': current_word})


@app.route('/api/add_space', methods=['POST'])
def add_space():
    global current_word, current_sentence, suggested_word
    if current_word.strip():
        # Use ORIGINAL word — suggestion was optional
        current_sentence = (current_sentence + ' ' + current_word).strip()
        current_word   = ''
        suggested_word = ''
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': '',
        'suggested': '', 'sentence': current_sentence
    })
    return jsonify({'sentence': current_sentence})


@app.route('/api/send_message', methods=['POST'])
def send_message():
    """
    Save and broadcast ANY user message.
    NO automatic bot reply — two-way real chat only.
    """
    global current_sentence
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or current_sentence or '').strip()
    if not text:
        return jsonify({'error': 'empty'}), 400

    sender     = data.get('sender', 'user-asl')
    sender_sid = data.get('sid', '')      # socket id from frontend
    msg = {
        'sender':    sender,
        'text':      text,
        'timestamp': time.strftime('%H:%M'),
        'skip_sid':  sender_sid,          # tells the sender's tab to skip this
    }
    chat_messages.append(msg)

    # Broadcast to ALL clients — frontend filters out its own copy via skip_sid
    socketio.emit('new_message', msg)

    # Clear ASL sentence after sending
    if sender == 'user-asl':
        current_sentence = ''
        socketio.emit('asl_update', {
            'letter': current_letter, 'word': current_word,
            'suggested': suggested_word, 'sentence': ''
        })

    return jsonify({'status': 'sent'})


@app.route('/api/messages')
def get_messages():
    return jsonify(chat_messages)


# ── SocketIO ──────────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('asl_update', {
        'letter': current_letter, 'word': current_word,
        'suggested': suggested_word, 'sentence': current_sentence,
    })
    # Send chat history to newly connected client
    for m in chat_messages[-50:]:
        emit('new_message', m)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_asl_model()
    print("\n🤟  ASLTalk starting…")
    print("   http://127.0.0.1:5000\n")
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, allow_unsafe_werkzeug=True)