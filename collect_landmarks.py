import cv2
import mediapipe as mp
import numpy as np
import os

DATA_DIR = "landmark_data"

labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

current_label = 0
counter = 0

os.makedirs(DATA_DIR, exist_ok=True)

for label in labels:
    os.makedirs(f"{DATA_DIR}/{label}", exist_ok=True)

print("Press SPACE to capture landmark")
print("Press N for next letter")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            lm_list = []

            for lm in hand_landmarks.landmark:
                lm_list.append(lm.x)
                lm_list.append(lm.y)

            cv2.putText(frame, labels[current_label], (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):
        if result.multi_hand_landmarks:
            np.save(
                f"{DATA_DIR}/{labels[current_label]}/{counter}.npy",
                np.array(lm_list)
            )
            counter += 1
            print("Saved:", counter)

    if key == ord('n'):
        current_label += 1
        counter = 0
        print("Next letter")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()