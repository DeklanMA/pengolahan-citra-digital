import cv2
import numpy as np


cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([66, 98, 100])
    upper_color = np.array([156, 232, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
