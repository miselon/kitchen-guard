from ultralytics import YOLO
import cv2
import math 
import numpy as np


# start webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)


##
# DETECTION
##

# model = YOLO("yolo-Weights/yolov8s.pt")

##
# SEGMENTATION
##

def draw_mask(image, mask_data, class_colors):

    for class_id, mask in enumerate(mask_data):

        # Convert class_id to a color
        color = class_colors[class_id]

        # Apply the mask to the image using the specified color
        image = cv2.addWeighted(image, 1, np.expand_dims(mask, axis=-1) * color, 0.5, 0)

    return image


model = YOLO("yolo-Weights/yolov8m-seg.pt")


while True:

    # _, img = cap.read()

    results = model(source="0", show=True)

    # for r in results:

    #     classes = r.names

    #     for mask in r.masks:

    #         print(mask.xy)
    #         break

            # img = cv2.addWeighted(img, 1, np.expand_dims(mask.xy, axis=-1) * (0, 0, 255), 0.5, 0)

        # for box in r.boxes:

        #     className = classes[int(box.cls[0])]

        #     if className not in ["cat", "dog", "person"]:
        #         continue

        #     # bounding box
        #     x1, y1, x2, y2 = box.xyxy[0]
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        #     # put box in cam
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        #     # confidence
        #     confidence = math.ceil((box.conf[0]*100))/100
        #     print("Confidence --->",confidence)

        #     # class name
        #     print("Class name -->", className)

        #     # object details
        #     org = [x1, y1]
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     fontScale = 1
        #     color = (255, 0, 0)
        #     thickness = 2

        #     cv2.putText(img, className, org, font, fontScale, color, thickness)

    # cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break


# cap.release()
# cv2.destroyAllWindows()