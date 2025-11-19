import cv2
import numpy as np
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Default Brush
draw_color = (0, 0, 255)  # RED
brush_thickness = 12
eraser_thickness = 50

prev_x, prev_y = 0, 0

# Button positions (x1, y1, x2, y2)
buttons = {
    "Red": (10, 10, 110, 110),
    "Blue": (120, 10, 220, 110),
    "Green": (230, 10, 330, 110),
    "Yellow": (340, 10, 440, 110),
    "Eraser": (450, 10, 600, 110),
    "Clear": (610, 10, 760, 110)
}

colors = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Yellow": (0, 255, 255),
}

def draw_buttons(img):
    for name, (x1, y1, x2, y2) in buttons.items():
        if name in colors:
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[name], -1)
        elif name == "Eraser":
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), -1)
        elif name == "Clear":
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        cv2.putText(img, name, (x1+10, y1+70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 0), 2)

def fingers_up(lmList):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb
    if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in tip_ids[1:]:
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    draw_buttons(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    lmList = []
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

        for id, lm in enumerate(hand.landmark):
            h, w, c = frame.shape
            lmList.append([id, int(lm.x * w), int(lm.y * h)])

    mode = ""  # To store current mode (for display)

    if lmList:
        finger_state = fingers_up(lmList)
        x, y = lmList[8][1], lmList[8][2]  # Index finger

        # Selection Mode (Index + Middle finger up)
        if finger_state[1] == 1 and finger_state[2] == 1:
            prev_x, prev_y = 0, 0
            mode = "selection"

            for name, (x1, y1, x2, y2) in buttons.items():
                if x1 < x < x2 and y1 < y < y2:
                    if name in colors:
                        draw_color = colors[name]
                    elif name == "Eraser":
                        draw_color = (0, 0, 0)
                    elif name == "Clear":
                        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Drawing Mode (only Index finger up)
        elif finger_state[1] == 1 and finger_state[2] == 0:
            mode = "drawing"

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            if draw_color == (0, 0, 0):
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, eraser_thickness)
            else:
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)

            prev_x, prev_y = x, y

    # Mode Display Box
    if mode == "selection":
        cv2.rectangle(frame, (10, 120), (360, 180), (0, 255, 255), 2)
        cv2.putText(frame, "SELECTION MODE", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # cv2.rectangle(frame, (10, 640), (360, 700), (0, 255, 255), 2)
        # cv2.putText(frame, "SELECTION MODE", (20, 685),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    elif mode == "drawing":
        cv2.rectangle(frame, (10, 120), (330, 180), (0, 255, 0), 2)
        cv2.putText(frame, "DRAWING MODE", (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # cv2.rectangle(frame, (10, 640), (330, 700), (0, 255, 0), 2)
        # cv2.putText(frame, "DRAWING MODE", (20, 685),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Merge canvas and frame
    grayCanvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(grayCanvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Canvas Pro", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
