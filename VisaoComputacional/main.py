import numpy as np
import argparse
import os
import cv2
from imutils.video import FPS

CONFIDENCE_MIN = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = "yolo-coco"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Caminho do arquivo de vídeo")
video_path = vars(ap.parse_args())['input']

print("[+] Carregando labels das classes treinadas...")
with open(os.path.sep.join([MODEL_BASE_PATH, 'coco.names'])) as f:
    labels = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
print("[+] Carregando o modelo YOLO treinado no COCO dataset...")
net = cv2.dnn.readNetFromDarknet(
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.cfg']),
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.weights']))

ln = [net.getUnconnectedOutLayers()[0] - 1]

vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
fps = FPS().start()
print("[+] Iniciando a captura de vídeo...")

fps_input = vs.get(cv2.CAP_PROP_FPS)

output_video_path = "D:\WorkSpace VSCode\VisaoComputacional/video_detectado.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps_input, (int(vs.get(3)), int(vs.get(4))))

while True:
    ret, frame = vs.read()

    if not ret:
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_outputs = net.forward([net.getUnconnectedOutLayersNames()[0]])

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_MIN and class_id in [0, 1, 2, 3]:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    new_ids = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_MIN, NMS_THRESHOLD)
    if len(new_ids) > 0:
        for i in new_ids.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color_picked = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_picked, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_picked, 2)

    cv2.imshow('frame', frame)

    output_video.write(frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

    fps.update()

fps.stop()
cv2.destroyAllWindows()
vs.release()
output_video.release()
