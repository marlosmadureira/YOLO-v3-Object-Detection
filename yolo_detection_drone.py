import cv2
import numpy as np
import pytesseract

# Configuração do pytesseract (necessário para Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carregar o modelo YOLOv3
model_cfg = 'cfg/yolov3.cfg'
model_weights = 'model/yolov3.weights'
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Carregar nomes das classes (COCO)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Defina a URL RTMP para o stream do drone
rtmp_url = 'rtmp://179.107.1.50/live/drone1'  # Substitua pela URL RTMP do drone
cap = cv2.VideoCapture(rtmp_url)

# Verificar se o stream RTMP foi aberto corretamente
if not cap.isOpened():
    print("Erro ao acessar o stream RTMP.")
    exit()


# Função para obter a saída da rede YOLO
def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Loop para processar cada frame do stream RTMP
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem do stream.")
        break

    # Pré-processamento do frame para YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Fazer a detecção de objetos
    outs = net.forward(get_outputs_names(net))

    # Processar os resultados
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filtrar para detectar somente as placas ou veículos com confiança acima de 0.5
            if confidence > 0.5 and classes[class_id] in ['car', 'vehicle']:  # Ajuste a classe conforme necessário
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maxima Suppression para remover caixas duplicadas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Recortar a região da placa
        placa = frame[y:y + h, x:x + w]

        # Aplicar OCR na placa para extrair o texto
        texto_placa = pytesseract.image_to_string(placa, config='--psm 8')

        # Desenhar a caixa delimitadora e exibir o texto da placa
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Placa: {texto_placa.strip()}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o frame ao vivo
    cv2.imshow('Detecção de Placas no Stream RTMP', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
