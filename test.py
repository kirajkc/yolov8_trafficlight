from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
# read webcam
cap = cv2.VideoCapture(0)
# visualize webcam
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame,classes = [9], imgsz = 256, vid_stride = 2)
        annotated_frames = results[0].plot()
        for i in results:
            if i.boxes:
                xywh = i.boxes.xywh.tolist()[0]
                x,y,w,h = xywh
                # roi = frame[y: y+h,x: x+w]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                l_red = np.array([0,120,70])
                u_red = np.array([10,255,255])
                l_green = np.array([45,100,50])
                u_green = np.array([75,255,255])
                l_yellow = np.array([20,100,100])
                u_yellow = np.array([30,255,255])


                mask_red = cv2.inRange(hsv, l_red, u_red)
                mask_green = cv2.inRange(hsv, l_green, u_green)
                mask_yellow = cv2.inRange(hsv, l_yellow, u_yellow)

                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)
                yellow_pixels = cv2.countNonZero(mask_yellow)

                if red_pixels > green_pixels and red_pixels > yellow_pixels:
                    print("red light")
                elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                    print("green light")
                elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                    print("yellow light")
                else:
                    print("not detected")

        cv2.imshow('frame', annotated_frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()