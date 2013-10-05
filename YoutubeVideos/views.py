from django.shortcuts import render
from YoutubeVideos.models import Video
import random

def index(request):

    labels = ["birthday", "parade", "picnic", "show", "sports", "wedding"]

    sampleVideos = []
    for label in labels:
        labelVideos = Video.objects.filter(label = label)
        labelVideos = random.sample(labelVideos, 3)
        sampleVideos.append(labelVideos)

    context = {"videoList": sampleVideos}

    return render(request, "index.html", context)