from django.shortcuts import render
from YoutubeVideos.models import Video
import Utility as util
import numpy as np
from sklearn.metrics import confusion_matrix
from confusionMatrix import ConfMatrix
from scipy.io import loadmat
from recognition import Base

trainVideoIndices = []
trainVideos = None

testVideoIndices = []
testVideos = None

gramma0 = 0
classifiers = []

def recognition(request):

    return render(request, "base.html")


def previewTrainData(request):

    global trainVideoIndices
    global testVideoIndices

    if request.method == "GET":
        num = request.GET["num"]
        num = int(num)

        if request.is_ajax():

            trainIndices, testIndices = Base.randomYoutubeIndices(num)
            allLabels = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/recognition/all_labels_Level0.pkl")

            trainVideosWithLabels = []
            for l in ["birthday", "parade", "picnic", "show", "sports", "wedding"]:
                tempVideos = []
                for v in trainIndices:
                    if allLabels[v] == l:
                        video = Video.objects.filter(indice = v)
                        tempVideos.append(video[0])
                trainVideosWithLabels.append(tempVideos)


            temp = []
            for videos in trainVideosWithLabels:
                temp.append(videos[:3])

            context = {"videoList": temp}


            trainVideoIndices = trainIndices
            testVideoIndices = testIndices
            return render(request, "videoList.html", context)


def trainSVM(request):
    global trainVideoIndices

    distanceOne = util.loadObject("/Users/GongLi/PycharmProjects/DomainAdaption/Distances/voc1000/All/GMM_ALL_Distance.pkl")
    distances = []
    distances.append(distanceOne)

    labels = loadmat("/Users/GongLi/PycharmProjects/DomainAdaption/labels.mat")['labels']

    Base.buildClassifiers(distances, labels, trainVideoIndices)


    # show training distsnce matrix
    trainDistance = distanceOne[np.ix_(trainVideoIndices, trainVideoIndices)]
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
    global testVideoIndices

    if request.method == "GET":

        if request.is_ajax():
            allLabels = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/recognition/all_labels_Level0.pkl")

            testVideosWithLabels = []
            for l in ["birthday", "parade", "picnic", "show", "sports", "wedding"]:
                tempVideos = []
                for v in testVideoIndices:
                    if allLabels[v] == l:
                        video = Video.objects.filter(indice = v)
                        tempVideos.append(video[0])
                testVideosWithLabels.append(tempVideos)

            temp = []
            for videos in testVideosWithLabels:
                temp.append(videos[:3])

            context = {"videoList": temp}

            return render(request, "videoList.html", context)

def testSVM(request):
    global testVideoIndices

    predictions = Base.predict(testVideoIndices)
    allLabels = util.loadObject("/Users/GongLi/PycharmProjects/VisualEventRecognitionDemo/YoutubeVideos/recognition/all_labels_Level0.pkl")
    originalLabels = [allLabels[i] for i in testVideoIndices]

    # Confusion stuff
    classLabels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]
    cm = ConfMatrix(confusion_matrix(originalLabels, predictions), classLabels)
    cm.gen_conf_matrix("confusion")

    # Accuracy
    correct = sum(1.0 * (np.array(predictions) == np.array(originalLabels)))
    context = {"nominator": int(correct),"denominator": len(originalLabels), "accuracy": correct / len(originalLabels)}


    return render(request, "predictionPresentation.html", context)

def index(request):

    context = {}
    return render(request, "index.html", context)








