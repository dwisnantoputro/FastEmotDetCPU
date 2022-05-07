# load json and create model

from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
from keras.models import load_model
from keras_layer_normalization import LayerNormalization
from sklearn.model_selection import train_test_split
import tensorflow as tf




x = np.load('./Datasets/FER-2013/modXtest.npy')
y = np.load('./Datasets/FER-2013/modytest.npy')

model_path = 'models/FER2013/fer2013-best.hdf5'

loaded_model = load_model(model_path, compile=False, custom_objects = {'LayerNormalization':LayerNormalization, 'tf':tf})


loaded_model.summary()

print("Loaded model from disk")

truey=[]
predy=[]

yhat= loaded_model.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

acc = (count/len(y))*100

np.save('./Datasets/FER-2013/truey', truey)
np.save('./Datasets/FER-2013/predy', predy)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")
