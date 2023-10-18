import sys
import os
import numpy as np
import json
import tensorflow as tf
from keras import layers, models, regularizers
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Defina hiperparâmetros ajustáveis
img_size = (312, 312)
batch_size = 32  # Reduzido para evitar potenciais problemas de OOM
epochs = 10
confidence_threshold = 0.2

# Ajustes para o caminho do diretório
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Defina o caminho para seu dataset e o tamanho da imagem que você deseja usar
dataset_path = os.path.join(script_dir, 'imagens')
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"O diretório {dataset_path} não foi encontrado.")c

# Crie geradores de dados para carregar e pré-processar as imagens
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Obtenha o número de classes e um mapeamento de índices para nomes de classes
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

# Salve o mapeamento de classe em um arquivo JSON
# Carregue o mapeamento de classe a partir do arquivo JSON
with open('class_names.json', 'r') as f:
    class_names = json.load(f)


try:
    model = load_model('meu_modelo_best.h5', compile=True)
except:
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3), kernel_regularizer=regularizers.l2(0.005)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Defina um callback para salvar o modelo com o melhor desempenho
save_filepath = 'meu_modelo_best.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=save_filepath,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Adicione um callback adicional para ajustar a taxa de aprendizado
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3)

# Callback para monitorar a taxa de aprendizado
class LrMonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f" Learning rate for epoch {epoch+1} is {lr:.5f}")

# Treine o modelo com os callbacks
model.fit(
    train_generator, 
    validation_data=validation_generator, 
    epochs=epochs, 
    callbacks=[checkpoint_callback, reduce_lr, LrMonitorCallback()]
)

# Salve o modelo no final do treinamento no formato .h5
model.save('meu_modelo_final.h5', save_format='h5')

def predict_image(img_path, confidence_threshold):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
        
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_idx]
    
    # Verificar se a previsão máxima é maior que o limiar de confiança
    if max(predictions[0]) < confidence_threshold:
        print(f'Objeto desconhecido - Classe prevista: {predicted_class_name}')
    else:
        print(f'Classe prevista: {predicted_class_name}')

# Lista de caminhos de imagens para previsão
image_paths = [os.path.join(script_dir, 'testeimg/teste4.jpg')] 

