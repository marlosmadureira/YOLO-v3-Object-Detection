import numpy as np
import cv2

# Parâmetros de confiança e supressão
confidenceThreshold = 0.5
NMSThreshold = 0.3

# Caminhos para o modelo YOLO
modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'model/yolov3.weights'

# Carregar rótulos
labelsPath = 'coco.names'
labels = open(labelsPath).read().strip().split('\n')

# Gerar cores aleatórias para as classes
np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Carregar a rede YOLO
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Obter as camadas de saída
outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i - 1] for i in net.getUnconnectedOutLayers()]

# Defina a URL RTMP do drone
rtmp_url = 'rtmp://179.107.1.50/live/drone1'  # Substitua pela URL do seu drone
video_capture = cv2.VideoCapture(rtmp_url)

# Variáveis para largura e altura do frame
(W, H) = (None, None)

# Loop principal para captura e processamento do vídeo
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Falha ao capturar o stream.")
        break

    frame = cv2.flip(frame, 1)  # Inverter o frame horizontalmente
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Pré-processar o frame para a entrada da rede
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)

    boxes = []
    confidences = []
    classIDs = []

    # Processar as saídas da rede
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Aplicar Non-Maxima Suppression para reduzir caixas sobrepostas
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if len(detectionNMS) > 0:
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"{text}")  # Adicione sua lógica de resposta aqui

    # Exibir o frame processado
    cv2.imshow('Detecção de Veículos no Stream do Drone', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
video_capture.release()
cv2.destroyAllWindows()
