
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
model = model_from_json(open("./model.json", "r").read())
model.load_weights('./model.h5')
face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    ret, image=cap.read()

    converted_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(converted_image)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0))
        roi_gray=converted_image[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))

        resized_image = cv2.resize(roi_gray, (48, 48))
        image_pixels = np.expand_dims(resized_image, axis=0)
        image_pixels = image_pixels / 255.0

        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])

        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]

        cv2.putText(image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    resized_image = cv2.resize(image, (1000, 700))
    cv2.imshow('Emotion', resized_image)
    if cv2.waitKey(10) == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()
