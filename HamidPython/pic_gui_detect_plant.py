import cv2
import numpy as np
from tkinter import Tk, filedialog


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(image):
    """Detect objects in the given image using YOLO."""
    height, width, _ = image.shape
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
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids, confidences

def main():
    
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(title="Select Plant Image ",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        print("No file selected.")
        return

    
    image = cv2.imread(file_path)
    if image is None:
        print("Error reading image.")
        return

    
    indexes, boxes, class_ids, confidences = detect_objects(image)

    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 0, 0)


            if label == "pottedplant":  
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    cv2.imshow("Plant Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
