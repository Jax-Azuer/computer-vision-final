import numpy as np
import cv2
import sys
import os
import gc
from dognet_train import Model

model = Model()
model.load_model(file_path = './model/dog_breed3.h5')
dog_cascade = cv2.CascadeClassifier('./dog_classifier.xml')

image = cv2.imread('./xxx.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dogs = dog_cascade.detectMultiScale(gray, 1.005, 50, 0, minSize=(100, 100))
if len(dogs) > 0:
    for (x, y, w, h) in dogs:
        dogID = model.face_predict(image)
        print("dog_breed: ", dogID)
        if dogID == 0:
            for (x, y, w, h) in dogs:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Cardigan',
                            (x+60, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2)
        elif dogID == 1:
            for (x, y, w, h) in dogs:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Chihuahua',
                            (x+60, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2)
        if dogID == 2:
            for (x, y, w, h) in dogs:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Labrador',
                            (x+60, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2)
        if dogID == 3:
            for (x, y, w, h) in dogs:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Siberian_husky',
                            (x+60, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2)
        if dogID == 4:
            for (x, y, w, h) in dogs:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Toy_poodle',
                            (x+60, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2)
        else:
            pass

    cv2.imshow('dog_detect', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()