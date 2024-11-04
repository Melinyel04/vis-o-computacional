import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicialização do MediaPipe para detecção das mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuração da câmera
cap = cv2.VideoCapture(0)

# Sensibilidade do movimento do cursor
cursor_sensibilidade = 2  # Ajuste conforme necessário
punho_fechado = False  # Variável para detectar se a mão está fechada

# Função para detectar se a mão está fechada (punho)
def detectar_punho(hand_landmarks):
    # Verifica a distância entre a ponta do dedo indicador e a base da mão
    # Se a distância é curta, a mão está fechada
    dedos_fechados = True
    for id_ponta, id_base in [(4, 0), (8, 5), (12, 9), (16, 13), (20, 17)]:
        ponta = hand_landmarks.landmark[id_ponta]
        base = hand_landmarks.landmark[id_base]
        dist = np.linalg.norm(np.array([ponta.x, ponta.y]) - np.array([base.x, base.y]))
        if dist > 0.1:  # Limite para considerar o dedo fechado
            dedos_fechados = False
            break
    return dedos_fechados

# Loop principal para detecção e execução de ações
with mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Erro ao capturar a câmera")
            break

        # Espelha e converte a imagem para RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a mão
        hand_results = hands.process(frame_rgb)

        # Verifica se há uma mão detectada
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Pega a posição da palma para movimentar o cursor
                pos_x = int(hand_landmarks.landmark[9].x * pyautogui.size().width)
                pos_y = int(hand_landmarks.landmark[9].y * pyautogui.size().height)
                pyautogui.moveTo(pos_x, pos_y, duration=0.1)

                # Detecta se a mão está fechada para simular um clique do mouse
                if detectar_punho(hand_landmarks):
                    if not punho_fechado:  # Clique apenas quando a mão estiver fechada
                        pyautogui.click()
                        punho_fechado = True
                        print("Clique do mouse simulado")
                else:
                    punho_fechado = False  # Reseta para detectar o próximo clique

                # Desenha a malha da mão
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Exibe a imagem processada com detecção dos pontos
        cv2.imshow('Controle com a Mão', frame)

        # Finaliza com a tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
