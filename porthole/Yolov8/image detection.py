from ultralytics import YOLO
import cv2
model = YOLO("pt/bestn.pt")

result = model.predict("182216_34674_4316.jpg",save = True, conf=0.5)
plots = result[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()