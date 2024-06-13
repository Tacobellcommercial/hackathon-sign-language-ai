import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

#CONFIG

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


#create hand landmarker instance for images

options = vision.HandLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task") #medipaipe hand landmarker task object
)

detector = vision.HandLandmarker.create_from_options(options) #create dector object

directories_to_loop = ["b_label", "c_label", "d_label", "e_label", "l_label", "i_love_you_label", "hello_label"]
labels_dictionary = {"b_label": 0, "c_label": 1, "d_label": 2, "e_label": 3, "l_label": 4, "i_love_you_label": 5, "hello_label": 6}

features = []
labels = []

for directory in directories_to_loop:
    print(directory)
    for image in os.listdir(f"image_labels/{directory}"):
        if image != ".ipynb_checkpoints":
            image_data = cv2.imread(f"image_labels/{directory}/{image}")
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data) #preprocess data to MP
            detection_result = detector.detect(mp_image) #detect hand
            
            
            features_to_append = []

            #get coordinates
            hand_landmarks_list = detection_result.hand_landmarks
            if len(hand_landmarks_list) > 0: #issue was that it was appending an empty list, which messed up np.array() shape
                for detected_hand in hand_landmarks_list:
                    for hand_point in detected_hand:
                        features_to_append.append([hand_point.x, hand_point.y, hand_point.z])
            
                features.append(features_to_append) #append hand features from single hand image
                labels.append(labels_dictionary[directory])

features = np.array(features) #convert to np array
labels = np.array(labels)

#features.shape = (105, 21, 3)
#105 objects, 21 arrays with x, y, z coordinates from landmark object


#NEURAL NETWORK CREATION 

X = features
y = labels

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#X_train shape = (73, 21, 3)


from tensorflow.keras.utils import to_categorical

y_train_categorical = to_categorical(y_train, 7)
y_test_categorical = to_categorical(y_test, 7)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(7, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy") #softmax with categorical_crossentropy

#softmax offers nuanced probability for our 7 label choices

early_stop = EarlyStopping(monitor="loss", patience=25, mode="min", verbose=1)
model.fit(x=X_train, y=y_train_categorical, validation_data=(X_test, y_test_categorical), epochs=280, batch_size=64, callbacks=[early_stop])

model.save("sign_language_7_labels.keras")