import cv2
import mediapipe as mp
import numpy as np
import joblib
import asyncio
import websockets
import json
import time
import threading

# ================= MODEL =================
model = joblib.load("landmark_model.pkl")

CLASSES = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z","del","nothing","space"
]

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# ================= STATE =================
stable_letter = ""
stable_start = 0
last_add = 0
word = ""
sentence = ""

STABLE_TIME = 0.7
COOLDOWN = 0.7

latest_data = {"letter":"","word":"","sentence":""}

# ================= PREPROCESS =================
def preprocess_landmarks(hand_landmarks):

    lm = []

    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    for p in hand_landmarks.landmark:
        lm.append(p.x - base_x)
        lm.append(p.y - base_y)

    max_val = max([abs(v) for v in lm])

    if max_val != 0:
        lm = [v / max_val for v in lm]

    return np.array(lm).reshape(1, -1)

# ================= CAMERA LOOP =================
def camera_loop():

    global latest_data, stable_letter, stable_start, last_add, word, sentence

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred_letter = ""

        if result.multi_hand_landmarks:

            print("HAND DETECTED")

            for hand in result.multi_hand_landmarks:

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                lm = preprocess_landmarks(hand)

                pred = model.predict(lm)

                if isinstance(pred[0], str):
                    pred_letter = pred[0]
                else:
                    pred_letter = CLASSES[int(pred[0])]

                print("Detected:", pred_letter)

        now_t = time.time()

        if pred_letter == stable_letter:

            if now_t - stable_start > STABLE_TIME:

                if now_t - last_add > COOLDOWN:

                    if pred_letter == "space":
                        sentence += word + " "
                        word = ""

                    elif pred_letter == "del":
                        word = word[:-1]

                    elif pred_letter not in ["","nothing"]:
                        word += pred_letter

                    last_add = now_t

        else:
            stable_letter = pred_letter
            stable_start = now_t

        latest_data = {
            "letter": pred_letter,
            "word": word,
            "sentence": sentence.strip()
        }

        cv2.imshow("ASL Camera", frame)
        cv2.waitKey(1)

# ================= WEBSOCKET =================
async def handler(websocket):

    print("Frontend connected")

    while True:

        try:
            await websocket.send(json.dumps(latest_data))
        except:
            print("Frontend disconnected")
            break

        await asyncio.sleep(0.03)

async def main():

    print("🚀 ASL Server Running → ws://127.0.0.1:8765")

    async with websockets.serve(handler, "127.0.0.1", 8765):
        await asyncio.Future()

# ================= START =================
if __name__ == "__main__":

    threading.Thread(target=camera_loop, daemon=True).start()
    asyncio.run(main())