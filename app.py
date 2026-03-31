"""
ASL Real-Time Communication System
Flask + SocketIO backend with MediaPipe + TensorFlow model integration
"""

import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from collections import deque
import threading

# ── Optional heavy imports (graceful fallback if not installed) ──────────────
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed – running in DEMO mode")

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("[WARN] TensorFlow/Keras not installed – running in DEMO mode")

from difflib import get_close_matches

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asl_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Model paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = 'asl_landmark_model.h5'
LABELS_PATH = 'labels.npy'
MEAN_PATH   = 'mean.npy'
STD_PATH    = 'std.npy'

# ── Global state ─────────────────────────────────────────────────────────────
model       = None
labels      = None
mean_vals   = None
std_vals    = None
mp_hands    = None
hands       = None
mp_drawing  = None

current_letter     = ''
current_word       = ''
current_sentence   = ''
suggested_word     = ''

# Smoothing: keep last N predictions
SMOOTHING_WINDOW = 10
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Letter hold timer (avoid rapid-fire same letter)
last_letter      = ''
last_letter_time = 0
BASE_COOLDOWN    = 1.2   # seconds before a NEW letter registers
REPEAT_COOLDOWN  = 3.3   # seconds before the SAME letter registers again

chat_messages    = []     # {sender, text, timestamp}
camera_active    = False
cap              = None
camera_lock      = threading.Lock()

# Demo cycling letters when running without model
DEMO_LETTERS = list("HELLO WORLD")
demo_index   = 0
demo_timer   = 0

# ── Model loading ─────────────────────────────────────────────────────────────
def load_asl_model():
    global model, labels, mean_vals, std_vals, mp_hands, hands, mp_drawing

    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print(f"[OK] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERR] Failed to load model: {e}")

    if os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"[OK] Labels loaded: {labels}")

    if os.path.exists(MEAN_PATH):
        mean_vals = np.load(MEAN_PATH)
        print("[OK] Mean loaded")

    if os.path.exists(STD_PATH):
        std_vals = np.load(STD_PATH)
        print("[OK] Std loaded")

    if MEDIAPIPE_AVAILABLE:
        mp_hands   = mp.solutions.hands
        hands      = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        mp_drawing = mp.solutions.drawing_utils
        print("[OK] MediaPipe Hands ready")

# ── Prediction logic ──────────────────────────────────────────────────────────
def extract_landmarks(frame_rgb):
    """Extract 21 hand landmarks — only x,y (42 floats) to match model input shape (None, 42)."""
    if not MEDIAPIPE_AVAILABLE or hands is None:
        return None, None
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for pt in lm.landmark:
        coords.extend([pt.x, pt.y])   # ← only x,y — NO z
    return np.array(coords, dtype=np.float32), lm


def predict_letter(landmarks):
    """Normalise landmarks and run model inference with adaptive thresholds."""
    if model is None or labels is None:
        return None
    try:
        # mean_vals / std_vals may be scalars (shape ()) or arrays — handle both
        norm  = (landmarks - mean_vals) / (float(std_vals) + 1e-8) if std_vals.ndim == 0 \
                else (landmarks - mean_vals) / (std_vals + 1e-8)
        inp   = norm.reshape(1, -1)          # shape → (1, 42)
        pred  = model.predict(inp, verbose=0)
        idx   = np.argmax(pred)
        conf  = float(pred[0][idx])
        label = str(labels[idx])

        # Smart adaptive threshold — tighter for visually similar letters
        threshold = 0.7
        if label in ['R', 'U', 'V']:
            threshold = 0.9
        elif label in ['A', 'S', 'E', 'T']:
            threshold = 0.85

        if conf < threshold:
            return None

        return label

    except Exception as e:
        print(f"[ERR] Prediction error: {e}")
        return None


def get_smoothed_letter():
    """Return most-common letter in the smoothing buffer."""
    if not prediction_buffer:
        return None
    from collections import Counter
    counts = Counter([l for l in prediction_buffer if l is not None])
    if not counts:
        return None
    best, freq = counts.most_common(1)[0]
    if freq / SMOOTHING_WINDOW > 0.7:
        return best
    return None


def demo_letter():
    """Cycle through demo letters when model is unavailable."""
    global demo_index, demo_timer
    now = time.time()
    if now - demo_timer > 1.5:
        demo_timer = now
        demo_index = (demo_index + 1) % len(DEMO_LETTERS)
    return DEMO_LETTERS[demo_index]


# ── Dictionary-based correction ───────────────────────────────────────────────
import sys

# Load a word list: use nltk if available, otherwise fall back to /usr/share/dict/words
WORDS = set()
try:
    import nltk
    try:
        from nltk.corpus import words as nltk_words
        WORDS = set(w.lower() for w in nltk_words.words())
    except LookupError:
        nltk.download('words', quiet=True)
        from nltk.corpus import words as nltk_words
        WORDS = set(w.lower() for w in nltk_words.words())
    print(f"[OK] Dictionary loaded via nltk ({len(WORDS):,} words)")
except Exception:
    # Fallback: system word list (Linux/macOS) or a compact built-in set
    dict_path = 'dictionary_word.txt'
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            WORDS = set(w.strip().lower() for w in f if w.strip().isalpha())
        print(f"[OK] Dictionary loaded from {dict_path} ({len(WORDS):,} words)")
    else:
        # Minimal hard-coded fallback so the function always works
        WORDS = {
            'hello', 'world', 'help', 'book', 'name', 'good', 'bad', 'love',
            'time', 'work', 'home', 'play', 'read', 'sign', 'hand', 'word',
            'apple', 'ball', 'cat', 'dog', 'eat', 'food', 'give', 'have',
            'jump', 'keep', 'like', 'make', 'need', 'open', 'push', 'quit',
            'run', 'stop', 'talk', 'used', 'very', 'want', 'yes', 'zero',
        }
        print("[WARN] No system dictionary found – using minimal built-in word list")


def correct_word(word):
    """Return closest dictionary match using difflib, or the word itself."""
    if not word:
        return word.upper()
    matches = get_close_matches(word.lower(), WORDS, n=1, cutoff=0.6)
    return matches[0].upper() if matches else word.upper()


# ── Sentence prediction ────────────────────────────────────────────────────────
SENTENCE_TEMPLATES = {
    "hi":        ["hi how are you", "hi nice to meet you", "hi good morning"],
    "hel":       ["hello how are you", "hello nice to meet you", "hello good to see you"],
    "hello":     ["hello how are you", "hello nice to meet you", "hello good to see you"],
    "how":       ["how are you", "how is everything", "how can i help you"],
    "how are":   ["how are you doing today", "how are you feeling", "how are things going"],
    "my na":     ["my name is", "my name is nice to meet you"],
    "my name":   ["my name is", "my name is what is yours"],
    "i am":      ["i am fine thank you", "i am happy to meet you", "i am doing well"],
    "i ne":      ["i need help", "i need water", "i need some time"],
    "i need":    ["i need help please", "i need water thank you", "i need more time"],
    "i wa":      ["i want to go", "i want help", "i want water please"],
    "i want":    ["i want to go home", "i want help please", "i want some water"],
    "tha":       ["thank you very much", "thank you so much", "thanks a lot"],
    "thank":     ["thank you very much", "thank you so much", "thanks for your help"],
    "goo":       ["good morning", "good afternoon", "good evening", "good night"],
    "good":      ["good morning how are you", "good afternoon", "good evening"],
    "ple":       ["please help me", "please come here", "please wait a moment"],
    "please":    ["please help me", "please come here", "please wait for me"],
    "whe":       ["where is the bathroom", "where are you going", "where do you live"],
    "where":     ["where is the bathroom", "where are you going", "where is the exit"],
    "wha":       ["what is your name", "what time is it", "what are you doing"],
    "what":      ["what is your name", "what time is it", "what are you doing today"],
    "ca":        ["can you help me", "can you repeat that", "can we talk"],
    "can":       ["can you help me please", "can you repeat that slowly", "can we meet tomorrow"],
    "are you":   ["are you okay", "are you ready to go", "are you sure about that"],
    "i lo":      ["i love you", "i love this", "i love spending time with you"],
    "i love":    ["i love you so much", "i love this place", "i love spending time here"],
    "see you":   ["see you later", "see you tomorrow", "see you soon take care"],
    "nice":      ["nice to meet you", "nice weather today", "nice to see you again"],
    "nice to":   ["nice to meet you", "nice to see you again", "nice to hear from you"],
    "take":      ["take care of yourself", "take your time", "take a break please"],
    "take ca":   ["take care", "take care of yourself", "take care and stay safe"],
    "i fee":     ["i feel happy today", "i feel tired", "i feel great thank you"],
    "i feel":    ["i feel happy today", "i feel a bit tired", "i feel great thanks"],
    "let me":    ["let me know", "let me help you", "let me think about this"],
    "sorry":     ["sorry i did not understand", "sorry for the trouble", "sorry i am late"],
    "sor":       ["sorry i did not understand", "sorry for the trouble", "sorry about that"],
    "no":        ["no problem at all", "no thank you", "no i am fine"],
    "yes":       ["yes of course", "yes i understand", "yes please go ahead"],
    "okay":      ["okay sounds good", "okay i understand", "okay let us do it"],
    "wait":      ["wait a moment please", "wait i will be right back", "wait for me please"],
    "help":      ["help me please", "help is on the way", "help i need assistance"],
    "come":      ["come here please", "come with me", "come back later"],
    "eat":       ["eat something healthy", "eat before you go", "eat with me please"],
    "water":     ["water please thank you", "water is very important"],
    "i do":      ["i do not understand", "i do not know", "i do not want to go"],
    "i don":     ["i do not understand", "i do not know what to do", "i do not feel well"],
    "time":      ["time is running out", "time to go home", "time flies"],
}

sentence_predictions = []   # holds current top-3 predictions

def get_sentence_predictions(sentence_so_far, word_so_far):
    """Match combined text against templates. Returns up to 3 suggestions."""
    combined = (sentence_so_far + ' ' + word_so_far).strip().lower()
    if not combined:
        return []
    best, best_len = [], 0
    for trigger, suggestions in SENTENCE_TEMPLATES.items():
        if combined.startswith(trigger) or trigger.startswith(combined):
            if len(trigger) > best_len:
                best_len = len(trigger)
                best = suggestions
    if not best:
        parts = combined.split()
        last = parts[-1] if parts else ''
        if len(last) >= 2:
            for trigger, suggestions in SENTENCE_TEMPLATES.items():
                if trigger.startswith(last):
                    best = suggestions
                    break
    return best[:3]


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
            success, frame = cap.read()
            if not success:
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
                            mp_drawing.DrawingSpec(color=(0,200,0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0,150,0), thickness=2)
                        )
                    prediction_buffer.append(predict_letter(landmarks))
                    detected_letter = get_smoothed_letter()
                else:
                    prediction_buffer.append(None)
            else:
                # Demo mode
                detected_letter = demo_letter()

            # ── Letter hold logic (adaptive cooldown) ────────────────────────
            now = time.time()
            if detected_letter and detected_letter != ' ':
                allow = False
                if detected_letter == last_letter:
                    # Same letter needs a longer gap to avoid accidental repeats
                    if now - last_letter_time > REPEAT_COOLDOWN:
                        allow = True
                else:
                    # Different letter only needs the base cooldown
                    if now - last_letter_time > BASE_COOLDOWN:
                        allow = True

                if allow:
                    current_letter   = detected_letter
                    last_letter      = detected_letter
                    last_letter_time = now
                    if detected_letter not in ('del', 'space', 'nothing'):
                        current_word += detected_letter
                        # Live dictionary-based suggestion
                        if len(current_word) > 2:
                            suggested_word = correct_word(current_word)
                        else:
                            suggested_word = current_word.upper()

            # ── Overlay HUD ──────────────────────────────────────────────────
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h-120), (w, h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

            def put(text, y, size=0.7, color=(255,255,255), bold=False):
                th = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display, text, (12, y), th, size, (0,0,0), 4 if bold else 2)
                cv2.putText(display, text, (12, y), th, size, color, 2 if bold else 1)

            put(f"Letter : {current_letter}",    h-95, 0.65, (100,255,100), True)
            put(f"Word   : {current_word}",      h-68, 0.60, (255,255,255))
            put(f"Suggest: {suggested_word}",    h-44, 0.60, (255,220,80))
            put(f"Sentence: {current_sentence}", h-18, 0.55, (180,220,255))

            # Update sentence predictions
            sentence_predictions = get_sentence_predictions(current_sentence, current_word)

            # Emit to frontend
            socketio.emit('asl_update', {
                'letter':      current_letter,
                'word':        current_word,
                'suggested':   suggested_word,
                'sentence':    current_sentence,
                'predictions': sentence_predictions,
            })

            ret, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

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
        'demo_mode':       (model is None or not MEDIAPIPE_AVAILABLE),
        'letter':          current_letter,
        'word':            current_word,
        'suggested':       suggested_word,
        'sentence':        current_sentence,
        'predictions':     sentence_predictions,
    })


@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_letter, current_word, current_sentence, suggested_word
    current_letter = current_word = current_sentence = suggested_word = ''
    prediction_buffer.clear()
    socketio.emit('asl_update', {
        'letter': '', 'word': '', 'suggested': '', 'sentence': ''
    })
    return jsonify({'status': 'ok'})


@app.route('/api/backspace', methods=['POST'])
def backspace():
    global current_word, suggested_word, current_sentence
    if current_word:
        current_word = current_word[:-1]
        suggested_word = correct_word(current_word) if len(current_word) > 2 else current_word.upper()
    elif current_sentence:
        # Remove last word from sentence
        words = current_sentence.strip().split()
        if words:
            current_word   = words[-1]
            current_sentence = ' '.join(words[:-1])
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': current_word,
        'suggested': suggested_word, 'sentence': current_sentence
    })
    return jsonify({'word': current_word, 'sentence': current_sentence})


@app.route('/api/apply_correction', methods=['POST'])
def apply_correction():
    global current_word, suggested_word
    if suggested_word:
        current_word = suggested_word
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': current_word,
        'suggested': suggested_word, 'sentence': current_sentence
    })
    return jsonify({'word': current_word})


@app.route('/api/add_space', methods=['POST'])
def add_space():
    global current_word, current_sentence, suggested_word
    if current_word.strip():
        # ── FIXED: always commit the RAW typed word, NOT the suggestion.
        # User must explicitly press "Apply Fix" if they want the suggestion.
        word_to_add      = current_word          # never auto-apply suggested_word
        current_sentence = (current_sentence + ' ' + word_to_add).strip()
        current_word     = ''
        suggested_word   = ''
    preds = get_sentence_predictions(current_sentence, '')
    socketio.emit('asl_update', {
        'letter': current_letter, 'word': '',
        'suggested': '', 'sentence': current_sentence,
        'predictions': preds,
    })
    return jsonify({'sentence': current_sentence})


@app.route('/api/apply_prediction', methods=['POST'])
def apply_prediction():
    """User chose a sentence prediction chip — replace sentence with it."""
    global current_sentence, current_word, suggested_word, sentence_predictions
    data = request.get_json()
    text = data.get('text', '').strip()
    if text:
        current_sentence     = text
        current_word         = ''
        suggested_word       = ''
        sentence_predictions = get_sentence_predictions(text, '')
        socketio.emit('asl_update', {
            'letter':      current_letter,
            'word':        '',
            'suggested':   '',
            'sentence':    current_sentence,
            'predictions': sentence_predictions,
        })
    return jsonify({'sentence': current_sentence})


@app.route('/api/send_message', methods=['POST'])
def send_message():
    global current_sentence, current_word, suggested_word, sentence_predictions
    data   = request.get_json()
    text   = (data.get('text') or '').strip()
    sender = data.get('sender', 'userA')   # 'userA' = ASL signer, 'userB' = keyboard user

    # ── If ASL side sends without pressing Space first,
    #    auto-commit the current raw word (NOT suggestion) into the sentence ──
    if not text and sender == 'userA':
        if current_word.strip():
            current_sentence = (current_sentence + ' ' + current_word).strip()
            current_word     = ''
            suggested_word   = ''
        text = current_sentence.strip()

    if not text:
        return jsonify({'error': 'empty'}), 400

    # Save and broadcast to ALL connected clients (both sides see it)
    msg = {'sender': sender, 'text': text, 'timestamp': time.strftime('%H:%M')}
    chat_messages.append(msg)
    socketio.emit('new_message', msg)   # ← both userA and userB receive this

    # Clear ASL state only when ASL side sends
    if sender == 'userA':
        current_sentence     = ''
        current_word         = ''
        suggested_word       = ''
        sentence_predictions = []
        socketio.emit('asl_update', {
            'letter': current_letter, 'word': '',
            'suggested': '', 'sentence': '', 'predictions': [],
        })

    return jsonify({'status': 'sent', 'msg': msg})


@app.route('/api/messages')
def get_messages():
    return jsonify(chat_messages)

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('asl_update', {
        'letter':      current_letter,
        'word':        current_word,
        'suggested':   suggested_word,
        'sentence':    current_sentence,
        'predictions': sentence_predictions,
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_asl_model()
    print("\n🤟 ASL Communication System starting…")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)