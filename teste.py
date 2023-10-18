import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("GPU disponível!")
else:
    print("GPU não disponível.")
import tensorflow as tf
print("Número de GPUs disponíveis: ", len(tf.config.experimental.list_physical_devices('GPU')))
