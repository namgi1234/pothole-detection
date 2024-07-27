from ultralytics import YOLO
import cv2
import math
# from gtts import gTTS
# from playsound import playsound
import os

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# text = "포트홀이 인식되었습니다. 서행하십시오."

# tts = gTTS(text=text, lang = 'ko')
# tts.save("alert.mp3")

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("best.pt")

# object classes
classNames = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
confidence_data = []

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)
            confidence_data.append(confidence)
            print(confidence_data)
            # class name
            cls = int(box.cls[0])
            print("포트홀이 인식되었습니다. 서행하십시오.")
            # playsound("alert.mp3")

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            tag = classNames[cls] + str(confidence)
            cv2.putText(img, tag  , org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()