__author__ = 'GongLi'

import os
from scipy.cluster.vq import *
import pickle
from numpy import *
import numpy as np
import subprocess as sub
import cv2

def loadObject(fileName):
    file = open(fileName, "rb")
    obj = pickle.load(file)
    return obj

# extract features
def process_image_dsift(imagename,resultname):

    img = cv2.imread(imagename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)

    if not kp:
        return

    # np.savetxt(resultname, des, delimiter=" ")
    file = open(resultname, "w")
    width, height = des.shape
    for i in range(width):
        for j in range(height):
            file.write(str(int(des[i][j])) +" ")
        file.write("\n")
    file.close()


# read video features and construct histograms
def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)
    else:
        result = None

    return result

def buildHistogramForVideo(pathToVideo):
    vocabulary = loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/ModeTwo/voc.pkl")
    size = len(vocabulary)
    frames = os.listdir(pathToVideo)

    stackOfHistogram = []
    for frame in frames:
        # build histogram for this frame
        completePath = pathToVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        print completePath

        frameFeatures = []
        for line in lines:
            data = line.split(" ")
            feature = data[4:-1]

            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            feature = normalizeSIFT(feature)
            frameFeatures.append(feature)

        frameFeatures = np.vstack(frameFeatures)
        print "Frame features size: " +str(frameFeatures.shape)

        codes, distance = vq(frameFeatures, vocabulary)

        histogram = np.zeros(size)
        for code in codes:
            histogram[code] += 1

        stackOfHistogram.append(histogram.reshape(1,size))

    result = np.vstack(stackOfHistogram)
    print "Histogram Size: " +str(result.shape)

    return result

def constructBaseKernels(kernel_type, kernel_params, D2):

    baseKernels = []

    for i in range(len(kernel_type)):

        for j in range(len(kernel_params)):

            type = kernel_type[i]
            param = kernel_params[j]

            if type == "rbf":
                baseKernels.append(math.e **(- param * D2))
            elif type == "lap":
                baseKernels.append(math.e **(- (param * D2) ** (0.5)))
            elif type == "id":
                baseKernels.append(1.0 / ((param * D2) ** (0.5) + 1))
            elif type == "isd":
                baseKernels.append(1.0 / (param * D2 + 1))

    return baseKernels

def C_EMD(feature1, feature2):

    if feature1.shape[0] > 349:
        feature1 = feature1[:350]
    if feature2.shape[0] > 349:
        feature2 = feature2[:350]

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])


    groundDistanceFile = open("ModeTwo/groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance")
    sub.call(["/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/ModeTwo/EarthMoverDistance"])

    # Read in EMD distance
    file = open("ModeTwo/result", "r").readlines()

    while True:
        try:
            os.remove("ModeTwo/groundDistance")
            break
        except:
            print "ground distance is not deleted properly"

    return float(file[0])





