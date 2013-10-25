__author__ = 'GongLi'

from django.shortcuts import render
from django.http import HttpResponse
import os
import Utility as util
import cv2
from scipy.cluster.vq import *


#global variables
frameExtractionProgress = 0
currentVideoName = ""
featureExtractionNum= 0

currentHistogram = None
histogramProgress = 0

classifyPro= 0

def index(request):

    context = {}
    return render(request, "mode2index.html", context)

# Extract frames
def getFrames(request):
    global frameExtractionProgress
    global currentVideoName
    frameExtractionProgress = 0

    if request.method == "GET":

        videoPath = request.GET["path"]
        videoPath = videoPath.split("\\")[-1]

        fullFolderPath = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/frames/" +videoPath[:-4]
        singleFolderName = videoPath[:-4]
        currentVideoName = singleFolderName

        if not os.path.exists(fullFolderPath):
            os.makedirs(fullFolderPath)

        videoPath = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/ModeTwo/demoVideos/"+videoPath
        if request.is_ajax():

            getFramesFromVideo(videoPath, fullFolderPath)
            frames = os.listdir(fullFolderPath)
            detailedFrames = ["/static/frames/"+singleFolderName+"/" + i for i in frames]
            context = {"frames": detailedFrames}
            return render(request, "frame.html", context)

def getFramesFromVideo(videoPath, folderNameToStoreFrames):
    global frameExtractionProgress

    cap = cv2.VideoCapture(videoPath)
    index = 1

    fps = int(cap.get(5))
    totalFrame = int(cap.get(7))
    frameNumber = 0
    totalNumberIndex = 0

    while True:
        ret, im = cap.read()
        frameNumber += 1
        totalNumberIndex += 1

        if frameNumber == fps:
            cv2.imwrite(folderNameToStoreFrames +"/"+ str(index)+".jpg", im)
            index += 1
            frameNumber = 0
            frameExtractionProgress = int((float(totalNumberIndex) / totalFrame) * 100)

        if totalNumberIndex == totalFrame:
            print "Video sampling is finished"
            break

    frameExtractionProgress = 100

def frameExtractProgress(request):

    global frameExtractionProgress
    return HttpResponse(str(frameExtractionProgress))

# Extract features from frames
def getFeatures(request):

    global featureExtractionNum
    global currentVideoName
    featureExtractionNum = 0

    extractVideoSIFT(currentVideoName)

    return HttpResponse("Extract SIFT features successfully!")

def extractVideoSIFT(videoName):
    global featureExtractionNum

    videoPath = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/frames/"+videoName
    frames = os.listdir(videoPath)


    featureFolder = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/features/" + videoName
    if not os.path.exists(featureFolder):
        os.makedirs(featureFolder)

    index = 0
    numberOfframes = len(frames)
    for frame in frames:
        framePath = videoPath +"/"+ frame
        featurePath = featureFolder +"/"+frame[:-4]+".sift"
        util.process_image_dsift(framePath, featurePath)

        featureExtractionNum = int((float(index) / numberOfframes) * 100)
        index += 1

    featureExtractionNum = 100

def featureExtractionProgress(request):
    global featureExtractionNum

    return HttpResponse(str(featureExtractionNum))

# Convert to histogram
def buildHistogramForVideo(pathToVideo):
    global histogramProgress
    histogramProgress = 0

    vocabulary = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/ModeTwo/voc.pkl")
    size = len(vocabulary)
    frames = os.listdir(pathToVideo)

    totalFrames = len(frames)
    counter = 0

    stackOfHistogram = []
    for frame in frames:
        # build histogram for this frame
        completePath = pathToVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        print completePath

        frameFeatures = []
        for line in lines:
            data = line.split(" ")
            feature = data[:-1]

            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            feature = util.normalizeSIFT(feature)
            frameFeatures.append(feature)

        frameFeatures = np.vstack(frameFeatures)
        print "Frame features size: " +str(frameFeatures.shape)

        codes, distance = vq(frameFeatures, vocabulary)

        histogram = np.zeros(size)
        for code in codes:
            histogram[code] += 1

        stackOfHistogram.append(histogram.reshape(1,size))

        counter += 1
        histogramProgress = int((float(counter) / totalFrames) * 100)

    result = np.vstack(stackOfHistogram)
    print "Histogram Size: " +str(result.shape)

    return result

def convertToHistogram(request):
    global currentVideoName
    global currentHistogram

    featurePath = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/features/" +str(currentVideoName)
    print featurePath

    currentHistogram = buildHistogramForVideo(featurePath)

    return HttpResponse("The shape of converted histogram is " +str(currentHistogram.shape))

def histogramProgressFun(request):
    global histogramProgress
    return HttpResponse(str(histogramProgress))



# Classify this user uploaded video
from YoutubeVideos import views
import numpy as np
from sklearn.svm import SVC


trainVideoIndices = []
trainVideos = []
gramma0 = 0
classifiers = None

def previewTrainVideos(request):
    global trainVideoIndices
    global trainVideos
    trainVideoIndices = []

    if request.method == "GET":
        num = request.GET["num"]
        num = int(num)

        if request.is_ajax():
            sampleVideos = views.getTrainVideos(num)
            for labelVideo in sampleVideos:
                for video in labelVideo:
                    trainVideoIndices.append(video.indice)

            trainVideos = sampleVideos

            if num > 10:
                temp = []
                for videos in trainVideos:
                    temp.append(videos[:3])

                context = {"videoList": temp}
            else:
                context = {"videoList": trainVideos}

            return render(request, "videoList.html", context)

def trainSVM(request):
    global trainVideoIndices
    global gramma0
    global classifiers

    distances = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/Distances/all_DistanceMatrix_Level0.pkl")
    labels = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/Distances/all_labels_Level0.pkl")

    trainDistance = distances[np.ix_(trainVideoIndices, trainVideoIndices)]
    trainLabels = []
    for indice in trainVideoIndices:
        trainLabels.append(labels[indice])

    # Train this SVM classifier
    distance = trainDistance ** 2
    gramma0 = 1.0 / np.mean(distance)
    kernel_params = []
    kernel_params.append(gramma0)

    baseKernel = util.constructBaseKernels(["rbf", "lap", "isd","id"], kernel_params, distance)

    classifiers = []
    for k in baseKernel:
        clf = SVC(kernel="precomputed")
        clf.fit(k, trainLabels)
        classifiers.append(clf)

    row, column = trainDistance.shape
    transferedDistance = []
    for i in range(row):
        temp = []
        for j in range(column):
            temp.append(int(trainDistance[i][j]))
        transferedDistance.append(temp)

    context = {"distance": transferedDistance}
    return render(request, "distanceTable.html", context)

def calculateDistanceToTrainingVideos(testHistogramMatrix, trainingVideosPath):
    global classifyPro


    trainHistograms = []
    for trainPath in trainingVideosPath:
        trainHistograms.append(util.loadObject(trainPath))

    numberOfTrainVideos = len(trainingVideosPath)
    distances = np.zeros((numberOfTrainVideos)).reshape((1, numberOfTrainVideos))
    for i in range(numberOfTrainVideos):
        distances[0][i] = util.C_EMD(testHistogramMatrix, trainHistograms[i])
        print str(distances[0][i])

        classifyPro = int((i / float(numberOfTrainVideos)) * 100)

    return distances


def classifyInputVideo(request):
    global classifyPro


    global currentHistogram
    global classifiers
    global gramma0
    global trainVideos

    trainVideoHistogramPath = []
    for labelVideo in trainVideos:
        for trainVideo in labelVideo:
            videoName = trainVideo.path[:-4]
            label = videoName.split("/")[0]
            videoName = videoName.split("/")[1]
            videoName = "Histogram_"+videoName+".pkl"

            fullPath = "/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/YoutubeCompressedDataHistogramOnly/"+label+"/" +videoName
            trainVideoHistogramPath.append(fullPath)

    # calculate distances & construct gram matrix
    dis = calculateDistanceToTrainingVideos(currentHistogram, trainVideoHistogramPath)
    kernel_params = []
    kernel_params.append(gramma0)
    dis = dis ** 2
    baseKernels = util.constructBaseKernels(["rbf", "lap", "isd", "id"], kernel_params, dis)

    # classify
    typeKernels = ["rbf", "lap", "isd", "id"]
    predictLabels = []
    for i in range(len(typeKernels)):
        clf = classifiers[i]
        predictLabels.append(clf.predict(baseKernels[i]))

    classifyPro = 100

    context = {"labels": predictLabels}
    return render(request, "predictResults.html", context)


def classifyProgress(request):
    global classifyPro

    return HttpResponse(str(classifyPro))












