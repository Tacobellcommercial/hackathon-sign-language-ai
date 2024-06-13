import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

detector = vision.HandLandmarker.create_from_options(options)

import tensorflow
model = tensorflow.keras.models.load_model("sign_language_7_labels.keras")

def draw_landmarks_on_image(frame, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    
    if len(hand_landmarks_list) > 0:
        
        image_to_return = np.copy(frame) #only copy if a hand is detected
        coordinates_to_return = []
        
        for hand_detected in hand_landmarks_list:
            
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_detected
            ])
            
            solutions.drawing_utils.draw_landmarks(
                image_to_return,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
            
            for hand_point in hand_detected: #loop through 21 coordinate items to get x, y, z array
                coordinates_to_return.append([hand_point.x, hand_point.y, hand_point.z])
            
            return True, image_to_return, np.array(coordinates_to_return)
    else:
        return False, False, False

        
cap = cv2.VideoCapture(0)

prediction_labels = {
    0:"Predicted: B", 
    1: "Predicted: C", 
    2: "Predicted: D", 
    3: "Predicted: E",
    4: "Predicted: L",
    5: "Predicted: I Love You!",
    6: "Predicted: Hello!",
}
while True:
    
    ret, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    success, image_to_return, coordinates_to_return = draw_landmarks_on_image(frame, detection_result)
    
    if success:
        coordinates_to_return = coordinates_to_return.reshape(1, 21, 3)
        prediction = np.argmax(model.predict(coordinates_to_return), axis=1)
        cv2.putText(image_to_return, prediction_labels[prediction[0]], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, ), 2, cv2.LINE_AA)
        cv2.imshow("frame", image_to_return)
    else:
        cv2.putText(frame, "Hand not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()