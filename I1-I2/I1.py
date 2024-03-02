import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # biru
    lower_color = np.array([66, 98, 100])
    upper_color = np.array([156, 232, 255])
    # merah
    # lower_color = np.array([0, 141, 38])
    # upper_color = np.array([93, 220, 255])
    # Hijau
    # lower_color = np.array([43, 102, 70])
    # upper_color = np.array([70, 238, 255])
    # kuning
    # lower_color = np.array([28, 175, 114])
    # upper_color = np.array([37, 255, 255])
    # #Ungu
    # lower_color = np.array([123, 130, 161])
    # upper_color = np.array([142, 205, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    print("-----Nilai Matrix Awal 3x3-----""\n", frame)
    print("-----Nilai Matrix result-----""\n", result)
    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
