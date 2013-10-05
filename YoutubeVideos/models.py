from django.db import models

class Video(models.Model):

    path = models.CharField(max_length=100)
    indice = models.IntegerField()
    label = models.CharField(max_length=20)

    def __unicode__(self):
        return self.path