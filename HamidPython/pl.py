import cv2
import numpy as np


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()


output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


cap = cv2.VideoCapture(0)


def detect_objects(image):

    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


    net.setInput(blob)
    outputs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []


    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                
                class_ids.append(class_id)
                confidences.append(float(confidence))  
                boxes.append([x, y, w, h])

    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids, confidences  

while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    
    indexes, boxes, class_ids, confidences = detect_objects(frame)

    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)

    
            if label == "pottedplant":
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    
    cv2.imshow("Webcam - Plant Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
