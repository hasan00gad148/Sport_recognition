import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Dense, BatchNormalization, ReLU, Dropout, Conv2D, MaxPooling2D, Flatten, Add#, Average, Concatenate, SpatialDropout2D
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split




physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("Invalid device or cannot modify virtual devices once initialized.")
  pass
print(physical_devices[0])


label_encoder = {"Basketball" : 0, "Football" : 1, "Rowing" : 2, "Swimming" : 3, "Tennis" : 4, "Yoga" : 5}

path = f".\\NN_DataSet\\Train\\"
imgsP1 = os.listdir(path)
TrainImgs = []
TrainLabel = []
# x = 325 , y = 420
for f in imgsP1: 
  img = cv2.imread(path+f"\\"+f)
  img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
  img = img.astype(np.float32)
  img /= 255
  TrainImgs.append(img)
  TrainLabel.append(label_encoder[f.split('_')[0]])
  # pdb.set_trace()

TrainImgs = np.array(TrainImgs)
TrainLabel = np.reshape(TrainLabel,(-1,1))
del imgsP1


print(TrainLabel.shape,"=========",TrainImgs.shape)


# path2 = f".\\NN_DataSet\\Test\\"
# imgsP2 = os.listdir(path2)
# TestImgs = []
# testID = []
# # x = 500 , y = 700
# for f in imgsP2:
#   img = cv2.imread(path2+f"\\"+f)
#   img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
#   img = img.astype(np.float32)
#   img /= 255
#   TestImgs.append(img)
#   testID.append(f)

# TestImgs = np.array(TestImgs)
# del imgsP2


# print(len(testID),"=========",TestImgs.shape)

X_train, X_test, y_train, y_test = train_test_split(TrainImgs, TrainLabel, test_size=0.2, random_state=16)

print(X_train.shape,"===",y_train.shape)



def createModel (inputSize = (256,256,3)):
  x = tf.keras.Input(shape=(inputSize))

# convolutional layers
  #  1st block
  convL1_1 = Conv2D( 32, 5, activation='linear', padding="same")(x)
  bnL1_1 = BatchNormalization()(convL1_1)
  reluL1_1 = ReLU()(bnL1_1)

  convL1_2 = Conv2D( 32, 5, activation='linear', padding="same")(reluL1_1)
  bnL1_2 = BatchNormalization()(convL1_2)
  reluL1_2 = ReLU()(bnL1_2)

  poolL1 = MaxPooling2D(pool_size=4 ,strides=4, padding='valid')(reluL1_2)


  #  2nd block
  convL2_1 = Conv2D( 32, 5, activation='linear', padding="same")(poolL1)
  bnL2_1 = BatchNormalization()(convL2_1)
  reluL2_1 = ReLU()(bnL2_1)

  convL2_2 = Conv2D( 32, 5, activation='linear', padding="same")(reluL2_1)
  bnL2_2 = BatchNormalization()(convL2_2)
  reluL2_2 = ReLU()(bnL2_2)

      # skip connection
  skipConn = Add()([reluL2_2, poolL1])
  poolL2 = MaxPooling2D(pool_size=4 ,strides=4, padding='valid')(skipConn)

# Flattened
  flattend = Flatten()(poolL2)

# fully connected layer
  #  layer1
  fc1 = Dense(256, activation='linear', use_bias=True, kernel_regularizer=L1L2(l1=0.01, l2=0.1),)(flattend)
  bn1 = BatchNormalization()(fc1)
  relu1 = ReLU()(bn1)
  #  layer2
  fc2 = Dense(32, activation='linear', use_bias=True, kernel_regularizer=L1L2(l1=0.1, l2=1))(relu1)
  bn2 = BatchNormalization()(fc2)
  relu2 = ReLU()(bn2)
  #  softmax
  softmax = Dense(6, activation='softmax', use_bias=True)(relu2)

  model_1 = Model(inputs=x, outputs=softmax)
  return model_1







model_1 = createModel()


print(model_1.summary())

model_1.compile(optimizer=Adam(learning_rate=0.001,)
,loss=SparseCategoricalCrossentropy()
            ,metrics=["accuracy"])



model_1.fit(X_train, y_train, batch_size=32, epochs=256)


















