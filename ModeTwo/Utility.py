__author__ = 'GongLi'

from PIL import Image
import os
from scipy.cluster.vq import *
import pickle
from numpy import *
import numpy as np
import subprocess as sub


def loadObject(fileName):
    file = open(fileName, "rb")
    obj = pickle.load(file)
    return obj

# extract features
def process_image_dsift(imagename,resultname,size=20,steps=10,force_orientation=False,resize=None):
    """ Process an image with densely sampled SIFT descriptors
        and save the results in a file. Optional input: size of features,
        steps between locations, forcing computation of descriptor orientation
        (False means all are oriented upwards), tuple for resizing the image."""

    im = Image.open(imagename).convert('L')
    if resize!=None:
        im = im.resize(resize)
    m,n = im.size

    if imagename[-3:] != 'pgm':
        #create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size/3.0
    x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(),y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame")

    if '/Users/GongLi/Dropbox/FYP/PythonProject/vlfeat/bin/maci64' not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep +'/Users/GongLi/Dropbox/FYP/PythonProject/vlfeat/bin/maci64'

    os.system(cmmd)
    # print 'processed', imagename, 'to', resultname

# read video features and construct histograms
def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)
    else:
        result = None

    return result

def readVideoData(pathOfSingleVideo, subSampling = 5):
    frames = os.listdir(pathOfSingleVideo)

    stackOfSIFTFeatures = []
    for frame in frames:
        completePath = pathOfSingleVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        for line in lines[1::subSampling]:
            data = line.split(" ")
            feature = data[4:]
            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            # normalize SIFT feature
            feature = normalizeSIFT(feature)
            stackOfSIFTFeatures.append(feature)

    return np.array(stackOfSIFTFeatures)

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
    # os.environ['PATH'] += os.pathsep + '/usr/local/bin'

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


    groundDistanceFile = open("groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance")
    sub.call([""])

    # Read in EMD distance
    file = open("result", "r").readlines()
    # os.remove("groundDistance")

    return float(file[0])


def calculateDistanceToTrainingVideos(testHistogramMatrix, trainingVideosPath):
    trainHistograms = []
    for trainPath in trainingVideosPath:
        trainHistograms.append(loadObject(trainPath))

    numberOfTrainVideos = len(trainingVideosPath)
    distances = np.zeros((numberOfTrainVideos))
    for i in range(numberOfTrainVideos):
        distances[0][i] = C_EMD(testHistogramMatrix, trainHistograms[i])

    return distances



