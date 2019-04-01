import tensorflow as tf
import numpy as np
import json

model_name = 'cnnrandomS1'
model = tf.keras.models.load_model('./models/'+model_name+'.h5')

#recompile for fo-r compatible weight initialization 
modelJSON = json.loads(model.to_json())

#rename GlorotUniform to glorot_uniform for DL4J
i = 0
for layer in modelJSON['config']['layers']:
    for attribute, value in layer['config'].items():
        if attribute == 'kernel_initializer':
            if value['class_name'] == 'GlorotUniform':
                modelJSON['config']['layers'][i]['config'][attribute]['class_name'] = 'glorot_uniform'
    i += 1           

directory = './export/'
#save the json
with open(directory+model_name+'.json', 'w') as f:
    json.dump(modelJSON, f)

#seperatedly save the mdoel weights
model.save_weights(directory+model_name+'_weights.h5')