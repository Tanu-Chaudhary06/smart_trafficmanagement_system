import cv2
import csv
import time
from datetime import datetime
from ultralytics import YOLO


MODEL_PATH = r"C:\Users\ASUS\runs\detect\runs\traffic_model_final7\weights\best.pt"
model = YOLO(MODEL_PATH)

print("Model Loaded Successfully")


# CAMERA / VIDEO FEEDS
caps = [
    cv2.VideoCapture("lane1.mp4"),
    cv2.VideoCapture("lane2.mp4"),
    cv2.VideoCapture("lane3.mp4"),
    cv2.VideoCapture("lane4.mp4")
]


# VEHICLE WEIGHTS

VEHICLE_WEIGHTS = {
    0: 2.0,   # Car
    1: 1.5,   # Auto
    2: 4.0,   # Bus
    3: 4.0,   # Truck
    4: 1.0,   # Bike
    5: 2.5,   # Van
    6: 10.0   # Ambulance
}


# CSV FILE


with open("traffic_report.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time",
        "Green Lane",
        "Lane1",
        "Lane2",
        "Lane3",
        "Lane4",
        "Green Time"
    ])



# SIGNAL VARIABLES

current_green = 0
next_green = 1
green_timer = 10

last_time = time.time()

print("SMART TRAFFIC SYSTEM STARTED")
print("Press Q to Quit")


# MAIN LOOP


while True:

    frames = []

    # Read all feeds
    for cap in caps:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()

        frame = cv2.resize(frame, (640, 360))
        frames.append(frame)

    lane_scores = [0, 0, 0, 0]


  
    # YOLO PROCESSING 
  

    for lane_no, frame in enumerate(frames):

        results = model(frame, verbose=False)

        score = 0

        if results[0].boxes is not None:

            boxes = results[0].boxes

            for i in range(len(boxes)):

                cls_id = int(boxes.cls[i])

                if cls_id in VEHICLE_WEIGHTS:
                    score += VEHICLE_WEIGHTS[cls_id]

        lane_scores[lane_no] = score

        # CLEAN FRAME 
        output = frame.copy()

        # TEXT ONLY
        signal = "GREEN" if lane_no == current_green else "RED"
        color = (0, 255, 0) if signal == "GREEN" else (0, 0, 255)

        cv2.putText(output,
                    f"Lane {lane_no+1} Score: {score:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(output,
                    f"Signal: {signal}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        frames[lane_no] = output


    # TIMER LOGIC


    current_time = time.time()
    green_timer -= (current_time - last_time)
    last_time = current_time


   
    # CHANGE SIGNAL

    if green_timer <= 0:

        current_green = next_green

        green_timer = 5 + (lane_scores[current_green] * 0.5)
        green_timer = max(10, min(60, green_timer))

        now = datetime.now().strftime("%H:%M:%S")

        with open("traffic_report.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                now,
                f"Lane {current_green+1}",
                lane_scores[0],
                lane_scores[1],
                lane_scores[2],
                lane_scores[3],
                round(green_timer, 1)
            ])

        # Next lane selection
        highest = -1

        for i in range(4):
            if i != current_green and lane_scores[i] > highest:
                highest = lane_scores[i]
                next_green = i


    # DISPLAY GRID

    top = cv2.hconcat([frames[0], frames[1]])
    bottom = cv2.hconcat([frames[2], frames[3]])

    grid = cv2.vconcat([top, bottom])

    cv2.putText(grid,
                f"Green Time: {green_timer:.1f}s",
                (450, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3)

    cv2.putText(grid,
                f"Next Lane: {next_green+1}",
                (450, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3)


    cv2.imshow("Smart Traffic Management System", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


for cap in caps:
    cap.release()

cv2.destroyAllWindows()