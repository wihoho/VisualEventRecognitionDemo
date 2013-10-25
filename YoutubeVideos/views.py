from django.shortcuts import render
from YoutubeVideos.models import Video
import random
import Utility as util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from confusionMatrix import ConfMatrix

trainVideoIndices = []
trainVideos = None

testVideoIndices = []
testVideos = None

gramma0 = 0
classifiers = []

def getTrainVideos(num):

    labels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]

    sampleVideos = []
    for label in labels:
        labelVideos = Video.objects.filter(label = label)
        labelVideos = random.sample(labelVideos, num)
        sampleVideos.append(labelVideos)

    return sampleVideos

def getTestVideos(num):
    global trainVideoIndices
    global trainVideos

    testVideos = []
    labels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    for i in range(len(labels)):
        label = labels[i]
        labelVideos = Video.objects.filter(label = label)
        trainLabelVideos = trainVideos[i]

        candiates = list(labelVideos)
        for j in trainLabelVideos:
            candiates.remove(j)

        testVideos.append(random.sample(candiates, num))

    return testVideos

def recognition(request):

    return render(request, "base.html")


def previewTrainData(request):

    global trainVideoIndices
    global trainVideos

    trainVideoIndices = []

    if request.method == "GET":
        num = request.GET["num"]
        num = int(num)

        if request.is_ajax():

            sampleVideos = getTrainVideos(num)
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

def selectTestVideos(request):
    global testVideos
    global testVideoIndices

    testVideoIndices = []
    if request.method == "GET":
        num = request.GET["num"]
        num = int(num)

        if request.is_ajax():
            testVideos = getTestVideos(num)
            for labelVideo in testVideos:
                for video in labelVideo:
                    testVideoIndices.append(video.indice)


            if num > 10:
                temp = []
                for videos in testVideos:
                    temp.append(videos[:3])

                context = {"videoList": temp}
            else:
                context = {"videoList": testVideos}

            return render(request, "videoList.html", context)

def testSVM(request):
    global trainVideoIndices
    global testVideoIndices
    global gramma0
    global classifiers

    distances = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/Distances/all_DistanceMatrix_Level0.pkl")
    labels = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/static/Distances/all_labels_Level0.pkl")

    testTotrainDistance = distances[np.ix_(testVideoIndices, trainVideoIndices)]
    transferredDistance = []
    row, column = testTotrainDistance.shape
    for i in range(row):
        temp = []
        for j in range(column):
            temp.append(testTotrainDistance[i][j])
        transferredDistance.append(temp)

    kernel_params = []
    kernel_params.append(gramma0)
    distance = testTotrainDistance ** 2
    baseKernels = util.constructBaseKernels(["rbf", "lap", "isd", "id"], kernel_params, distance)

    typeKernels = ["rbf", "lap", "isd", "id"]
    predictLabels = []
    for i in range(len(typeKernels)):
        clf = classifiers[i]
        predictLabels.append(clf.predict(baseKernels[i]))

    originalLabels = []
    for i in testVideoIndices:
        originalLabels.append(labels[i])


    # videos = Video.objects

    accuracies = []
    for prediction in predictLabels:
        correct = sum(1.0 * (prediction == originalLabels))
        accuracy = correct / len(originalLabels)
        accuracies.append(accuracy)

        # for indice in range(0, len(originalLabels), 1):
        #     if originalLabels[indice] == prediction[indice]:
        #         videoIndice = testVideoIndices[indice]
        #         video = videos.filter(indice = videoIndice)
        #         print video



    # Confusion stuff
    classLabels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    for i in range(len(typeKernels)):
        cm = ConfMatrix(confusion_matrix(originalLabels, predictLabels[i]), classLabels)
        cm.gen_conf_matrix(typeKernels[i])

    context = {"originalLabels": originalLabels, "predictLabels": predictLabels, "accuracies": accuracies, "distance":transferredDistance}


    # print out correct labelled videos






    return render(request, "predictionPresentation.html", context)

def index(request):

    context = {}
    return render(request, "index.html", context)








