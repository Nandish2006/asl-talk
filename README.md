# 🤟 ASLTalk — Real-Time ASL Communication System

A full-stack Flask web app that reads American Sign Language hand gestures from
your webcam and converts them into text through a WhatsApp-style chat interface.

---

## 📁 Project Structure

```
asl_app/
├── app.py                    ← Main Flask + SocketIO backend
├── requirements.txt          ← Python dependencies
├── asl_landmark_model.h5     ← Your trained Keras model  ← ADD THIS
├── labels.npy                ← Class labels              ← ADD THIS
├── mean.npy                  ← Feature mean              ← ADD THIS
├── std.npy                   ← Feature std               ← ADD THIS
├── templates/
│   └── index.html            ← Split-screen UI
└── static/
    ├── style.css             ← WhatsApp-style CSS
    └── script.js             ← SocketIO + controls
```

---

## ⚙️ Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If TensorFlow install is slow, use the CPU-only variant:
> ```bash
> pip install tensorflow-cpu
> ```

### 3. Add your model files
Place these four files in the `asl_app/` root directory:
- `asl_landmark_model.h5`
- `labels.npy`
- `mean.npy`
- `std.npy`

> **Without model files:** The app runs in **Demo Mode** — it cycles through
> letters so you can test the UI without a trained model.

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🎮 How to Use

| Action | Control |
|--------|---------|
| Sign a letter | Show hand to webcam |
| Add letter to word | Just keep signing (auto-detected) |
| Confirm word → sentence | Click **␣ Space** or press `Space` |
| Remove last letter | Click **⌫ Backspace** or press `Backspace` |
| Apply autocorrect | Click **✓ Apply Fix** |
| Clear everything | Click **✕ Clear All** |
| Send ASL sentence to chat | Click **Send Sentence to Chat →** or press `Enter` |
| Type manually | Use the chat input box on the right |

---

## 🧠 Model Integration Details

The backend (`app.py`) expects:

1. **MediaPipe** extracts 21 hand landmarks → 63 floats `[x, y, z] × 21`
2. **Normalise**: `(landmarks − mean) / (std + 1e-8)`
3. **Model input shape**: `(1, 63)`
4. **Model output**: softmax over number of ASL classes
5. **Confidence threshold**: 0.60 (adjustable in `app.py`)
6. **Smoothing**: Majority vote over last 10 frames

---

## 🔌 Real-Time Communication

- **Flask-SocketIO** pushes ASL detection updates to the browser every frame
- **AJAX POST** endpoints handle button actions
- Chat messages are stored in-memory (restart clears history)

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not showing | Allow camera permission in browser; check another app isn't using it |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Model not loading | Ensure `.h5` / `.npy` files are in the same folder as `app.py` |
| Port 5000 in use | Change `port=5000` in `app.py` to e.g. `5001` |
| Slow on CPU | Reduce `SMOOTHING_WINDOW` or lower camera resolution |

---

## 🛠 Configuration (app.py)

```python
SMOOTHING_WINDOW = 10     # frames for majority-vote smoothing
LETTER_HOLD_SEC  = 1.2    # seconds before same letter re-registers
# Confidence threshold in predict_letter():
if conf < 0.6: return None
```
