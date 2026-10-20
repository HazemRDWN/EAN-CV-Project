import os
import numpy as np
import cv2 

def loadImages(folder):
    images = []

    for fileName in os.listdir(folder):

        image = cv2.imread(os.path.join(folder, fileName), cv2.IMREAD_GRAYSCALE)
        if image is not None: 
            images.append(image)
            print(f"Loaded {fileName} image\n")

    os.system('cls')

    return images

def loadTargets(folder):
    targets = []

    for fileName in os.listdir(folder):

        image = cv2.imread(os.path.join(folder, fileName), cv2.IMREAD_GRAYSCALE)
        if image is not None: 
            targets.append(int(fileName[22]))
            print(f"Loaded {fileName} target\n")

    os.system('cls')

    return targets

xTrainRaw = (loadImages('MAHDBase_TrainingSet/Part01') + loadImages('MAHDBase_TrainingSet/Part02') + 
          loadImages('MAHDBase_TrainingSet/Part03') + loadImages('MAHDBase_TrainingSet/Part04') + 
          loadImages('MAHDBase_TrainingSet/Part05') + loadImages('MAHDBase_TrainingSet/Part06') +
          loadImages('MAHDBase_TrainingSet/Part07') + loadImages('MAHDBase_TrainingSet/Part08') + 
          loadImages('MAHDBase_TrainingSet/Part09') + loadImages('MAHDBase_TrainingSet/Part10') + 
          loadImages('MAHDBase_TrainingSet/Part11') + loadImages('MAHDBase_TrainingSet/Part12'))

yTrainRaw = (loadTargets('MAHDBase_TrainingSet/Part01') + loadTargets('MAHDBase_TrainingSet/Part02') + 
          loadTargets('MAHDBase_TrainingSet/Part03') + loadTargets('MAHDBase_TrainingSet/Part04') + 
          loadTargets('MAHDBase_TrainingSet/Part05') + loadTargets('MAHDBase_TrainingSet/Part06') +
          loadTargets('MAHDBase_TrainingSet/Part07') + loadTargets('MAHDBase_TrainingSet/Part08') + 
          loadTargets('MAHDBase_TrainingSet/Part09') + loadTargets('MAHDBase_TrainingSet/Part10') + 
          loadTargets('MAHDBase_TrainingSet/Part11') + loadTargets('MAHDBase_TrainingSet/Part12'))

xTestRaw = (loadImages('MAHDBase_TestingSet/Part01') + loadImages('MAHDBase_TestingSet/Part02'))

yTestRaw = (loadTargets('MAHDBase_TestingSet/Part01') + loadTargets('MAHDBase_TestingSet/Part02'))

xTrainNP = np.array(xTrainRaw)
yTrainNP = np.array(yTrainRaw)

xTestNP = np.array(xTestRaw)
yTestNP = np.array(yTestRaw)

print("Saving data..\n")

np.savez_compressed('mahdbase.npz', xTrain = xTrainNP, yTrain = yTrainNP, xTest = xTestNP, yTest = yTestNP)

print("Data saved in mahdbase.npz")

print("_________________________Training Data_________________________\n")
print(xTrainNP[:2])
print("_____Training Targets_____\n")
print(yTrainNP[:10], "\n")

print("_________________________Testing Data_________________________\n")
print(xTestNP[:2])
print("_____Testing Targets_____\n")
print(yTestNP[:10], "\n")