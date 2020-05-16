# preliminary results written and accomplished yesterday (in the middle of updating the code so that it is concise and make sense)
# ... also generating more data in order to get more accurate results. Using ParaView in Render Window mode and excel spreadsheet
# , one can select only the points they want to see in the render window (utilizing the selection tool widget) and generate
# a list of pIds that represent corners in the point cloud. Using the function below, you can pass the polydata and the list
# of point ids to update as corners then proceed to the neural network for training. These are the steps I took, and are 
# continuing to develop. This will eventually go back into the YDLidarSLAM project for Simultatenous Localization and mapping.

import os
import numpy as np
import vtk
from vtk.util import numpy_support as ns
from glob import glob

def tagCorners(aList, polydata, filename):
    corners = np.zeros(505).reshape(-1, 1)
    for pId in aList:
        corners[pId] = 1
    
    c = ns.numpy_to_vtk(corners)
    c.SetName("Corners")
    
    polydata.GetPointData().AddArray(c)
    
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName("/home/kdean/ydlidar_clouds/tagged_clouds/" + os.path.basename(filename))
    w.SetInputData(polydata)
    w.Write()

def ReadVTP(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader

X = []
y = []

files = glob("/home/kdean/ydlidar_clouds/tagged_clouds/*.vtp")
files.sort()

for i in files:
    reader = ReadVTP(i)
    vtp = reader.GetOutput()
    
    pts = list(map(lambda i: vtp.GetPoint(i), \
        range(vtp.GetNumberOfPoints())))
    
    c = list(ns.vtk_to_numpy(vtp.GetPointData().GetArray("Corners")))
    
    X.append(pts[0:504])
    y.append(c[0:504])

X = np.asarray(X)
y = np.asarray(y).reshape(-1, 504, 1)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate

# neural network
filters = 2
kernel = 2
pooling = 2
sampling = 2

inputs = Input(shape=(None, 3))

#############################################################################

c1 = Conv1D(filters * 16, kernel, activation="relu", padding="same")(inputs)
m1 = MaxPooling1D(2)(c1)

middle1 = Conv1D(filters * 32, kernel, activation="relu", padding="same")(m1)

up1 = UpSampling1D(2)(middle1)
concat1 = concatenate([up1, c1])
c2 = Conv1D(filters * 16, kernel, activation="relu", padding="same")(concat1)

#############################################################################

c3 = Conv1D(filters * 8, kernel, activation="relu", padding="same")(c2)
m2 = MaxPooling1D(2)(c3)

middle2 = Conv1D(filters * 16, kernel, activation="relu", padding="same")(m2)

up2 = UpSampling1D(2)(middle2)
concat2 = concatenate([up2, c3])
c4 = Conv1D(filters * 8, kernel, activation="relu", padding="same")(concat2)

#############################################################################

c5 = Conv1D(filters * 4, kernel, activation="relu", padding="same")(c4)
m3 = MaxPooling1D(2)(c5)

middle3 = Conv1D(filters * 8, kernel, activation="relu", padding="same")(m3)

up3 = UpSampling1D(2)(middle3)
concat3 = concatenate([up3, c5])
c6 = Conv1D(filters * 4, kernel, activation="relu", padding="same")(concat3)

#############################################################################

c7 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(c6)
m4 = MaxPooling1D(2)(c5)

middle4 = Conv1D(filters * 4, kernel, activation="relu", padding="same")(m4)

up4 = UpSampling1D(2)(middle3)
concat4 = concatenate([up3, c5])
c8 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(concat4)

#############################################################################

c9 = Conv1D(1, 1)(c8)

model = Model(inputs=inputs, outputs=c9)
model.compile(loss=Huber(), optimizer=Adam(lr=1e-4), metrics=['accuracy'])
model.summary()

callback_list = [ModelCheckpoint("best-model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]

model.fit(X, y, validation_split=0.2, epochs=500, batch_size=4, callbacks=callback_list)


files = glob("/home/kdean/ydlidar_clouds/data/*.vtp")
files.sort()

for aFile in files:
    print(aFile)
    reader = ReadVTP(aFile)
    vtp = reader.GetOutput()
    
    pts = list(map(lambda i: vtp.GetPoint(i), \
        range(vtp.GetNumberOfPoints())))
    
    X = np.array(pts[0:504]).reshape(-1, 504, 3)
    
    p = model.predict(X)
    p = list(p.reshape(504))
    p.append(0.0)
    
    for i in range(len(p)):
        if p[i] < 0.0: p[i] = 0.00005
    
    pc = ns.numpy_to_vtk(np.asarray(p))
    pc.SetName("Prediction")
    
    vtp.GetPointData().AddArray(pc)
    
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(aFile)
    w.SetInputData(vtp)
    w.Update()
