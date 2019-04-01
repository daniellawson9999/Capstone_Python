#programming for exporting keras models to tflite
#run inside of google colab
import tensorflow as tf
model_name = 'cnnfixedrightL'
keras_file = './models/' + model_name + '.h5'
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
tflite_model
open('./export/'+ model_name+'.tflite', 'wb').write(tflite_model)
