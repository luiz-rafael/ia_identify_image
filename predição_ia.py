import os
import sys
import numpy as np
import json
from keras.preprocessing import image
from keras.models import load_model

# Ajustes para o caminho do diretório
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Defina o caminho para seu dataset e o tamanho da imagem que você deseja usar
img_size = (312, 312)

# Carregar o modelo treinado
model_path = os.path.join(script_dir, 'meu_modelo_final.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"O modelo {model_path} não foi encontrado.")
    
model = load_model(model_path)

# Carregue o mapeamento de índices para nomes de classes
with open('IA TESTE\class_names.json', 'r') as f:
    class_names = json.load(f)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_idx]
    
    # Definir um limiar de confiança
    confidence_threshold = 0.6  # Você pode ajustar esse valor conforme necessário

    # Verificar se a previsão máxima é maior que o limiar de confiança
    if max(predictions[0]) < confidence_threshold:
        print('Objeto desconhecido')
    else:
        print(f'Classe prevista: {predicted_class_name}')

# Lista de caminhos de imagens para previsão
image_paths = [os.path.join(script_dir, 'testeimg/teste4.jpg')] 

# Teste a função de previsão com o caminho para uma nova imagem
for img_path in image_paths:
    predict_image(img_path)
    