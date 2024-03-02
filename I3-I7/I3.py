import cv2
import numpy as np

cam = cv2.VideoCapture('car.mp4')

car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow('video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
