import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('Trump.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No faces found")
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
