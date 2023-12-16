import numpy as np
import argparse
import os
import cv2
from imutils.video import FPS

# Constantes do modelo
CONFIDENCE_MIN = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = "yolo-coco"

# Receber os argumentos para o script
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Caminho do arquivo de vídeo")
video_path = vars(ap.parse_args())['input']

# Carregar as classes do modelo COCO
print("[+] Carregando labels das classes treinadas...")
with open(os.path.sep.join([MODEL_BASE_PATH, 'coco.names'])) as f:
    labels = f.read().strip().split("\n")

    # Gerar cores únicas para cada label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Carregar o modelo treinado YOLO (c/ COCO dataset)
print("[+] Carregando o modelo YOLO treinado no COCO dataset...")
net = cv2.dnn.readNetFromDarknet(
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.cfg']),
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.weights']))

# Extrair layers não conectados da arquitetura YOLO
ln = [net.getUnconnectedOutLayers()[0] - 1]

# Iniciar a captura de vídeo
vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
fps = FPS().start()
print("[+] Iniciando a captura de vídeo...")

# Obter a taxa de quadros do vídeo de entrada
fps_input = vs.get(cv2.CAP_PROP_FPS)

# Definir as configurações do vídeo de saída
output_video_path = "D:/Documentos/VisaoComputacional/video_detectado.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps_input, (int(vs.get(3)), int(vs.get(4))))

# Iterar sobre os frames do vídeo
while True:
    ret, frame = vs.read()

    # Verificar se o vídeo chegou ao fim
    if not ret:
        break

    # Capturar a largura e altura do frame
    (H, W) = frame.shape[:2]

    # Construir um container blob e fazer uma passagem (forward) na YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Fazer uma passagem (forward) na YOLO
    layer_outputs = net.forward([net.getUnconnectedOutLayersNames()[0]])

    # Criar listas com boxes, nível de confiança e ids das classes
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrar pelo threshold da confiança e classes específicas
            if confidence > CONFIDENCE_MIN and class_id in [0, 1, 2, 3]:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Eliminar ruído e redundâncias aplicando non-maxima suppression
    new_ids = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_MIN, NMS_THRESHOLD)
    if len(new_ids) > 0:
        for i in new_ids.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Plotar retângulo e texto das classes detectadas no frame atual
            color_picked = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_picked, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_picked, 2)

    # Exibir o frame atual
    cv2.imshow('frame', frame)

    # Escrever o frame no vídeo de saída
    output_video.write(frame)

    # Sair, caso seja pressionada a tecla ESC
    c = cv2.waitKey(1)
    if c == 27:
        break

    # Atualizar o fps
    fps.update()

# Eliminar processos e janelas
fps.stop()
cv2.destroyAllWindows()
vs.release()
output_video.release()
