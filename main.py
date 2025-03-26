import cv2
import mediapipe as mp
import pyautogui
import time
import tkinter as tk
from PIL import Image, ImageTk
import random

def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1

    return cnt

def update_object_position(count):
    global obj_x, obj_y
    step_size = 20  
    
    if count == 1:
        obj_x += step_size
    elif count == 2:
        obj_x -= step_size
    elif count == 3:
        obj_y -= step_size
    elif count == 4:
        obj_y += step_size
    
    obj_x = max(0, min(canvas_width - obj_size, obj_x))
    obj_y = max(0, min(canvas_height - obj_size, obj_y))
    
    canvas.coords(obj, obj_x, obj_y, obj_x + obj_size, obj_y + obj_size)
    check_collisions(obj_x, obj_y)
    check_win_condition(obj_x, obj_y)

def check_collisions(x, y):
    for obs in obstacles:
        if x < obs[0] + obstacle_size and x + obj_size > obs[0] and y < obs[1] + obstacle_size and y + obj_size > obs[1]:
            canvas.create_text(canvas_width / 2, canvas_height / 2, text="Game Over", font=("Arial", 24), fill="red")
            reset_game()
            return True
    return False

def check_win_condition(x, y):
    if x >= end_x and y >= end_y:
        canvas.create_text(canvas_width / 2, canvas_height / 2, text="You Win!", font=("Arial", 24), fill="green")
        reset_game()

def generate_obstacles(num_obstacles):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            x = random.randint(0, canvas_width - obstacle_size)
            y = random.randint(0, canvas_height - obstacle_size)
            if (x, y) != (obj_x, obj_y) and (x, y) != (end_x, end_y):
                obstacles.append((x, y))
                break
    return obstacles

def main_loop():
    global prev, start_init, start_time
    end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    cnt = 0
    if res.multi_hand_landmarks:
        hand_keyPoints = res.multi_hand_landmarks[0]
        cnt = count_fingers(hand_keyPoints)

        if not (prev == cnt):
            if not start_init:
                start_time = time.time()
                start_init = True
            elif (end_time - start_time) > 0.2:
                update_object_position(cnt)
                prev = cnt
                start_init = False

        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

    img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    img = ImageTk.PhotoImage(image=Image.fromarray(img))
    label_video.img = img
    label_video.config(image=img)

    root.after(1, main_loop)

def start_game():
    global obj_x, obj_y, obstacles, start_init, prev, start_time
    obj_x, obj_y = 0, 0
    canvas.coords(obj, obj_x, obj_y, obj_x + obj_size, obj_y + obj_size)
    start_init = False
    prev = -1
    start_time = time.time()
    canvas.delete("obstacle")
    obstacles = generate_obstacles(num_obstacles)
    for obs in obstacles:
        canvas.create_rectangle(obs[0], obs[1], obs[0] + obstacle_size, obs[1] + obstacle_size, fill="red", tags="obstacle")

def reset_game():
    start_game()

cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

root = tk.Tk()
root.title("Finger Movement Game")
root.geometry("700x600")

frame_video = tk.Frame(root)
frame_video.pack(padx=10, pady=10)

label_video = tk.Label(frame_video)
label_video.pack()

canvas_width, canvas_height = 400, 400
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white", highlightthickness=2, highlightbackground="gray")
canvas.pack(padx=10, pady=10)

obj_size = 20
obj_x, obj_y = 0, 0
obj = canvas.create_rectangle(obj_x, obj_y, obj_x + obj_size, obj_y + obj_size, fill="blue")

end_size = 20
end_x, end_y = canvas_width - end_size, canvas_height - end_size
canvas.create_rectangle(end_x, end_y, end_x + end_size, end_y + end_size, fill="green")

obstacle_size = 30
num_obstacles = 10
obstacles = generate_obstacles(num_obstacles)
for obs in obstacles:
    canvas.create_rectangle(obs[0], obs[1], obs[0] + obstacle_size, obs[1] + obstacle_size, fill="red", tags="obstacle")

btn_start = tk.Button(root, text="Start", command=start_game, bg="lightblue", font=("Arial", 14), padx=10, pady=5)
btn_start.pack(side=tk.LEFT, padx=10, pady=10)

btn_reset = tk.Button(root, text="Reset", command=reset_game, bg="lightcoral", font=("Arial", 14), padx=10, pady=5)
btn_reset.pack(side=tk.RIGHT, padx=10, pady=10)

start_init = False
prev = -1

main_loop()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
