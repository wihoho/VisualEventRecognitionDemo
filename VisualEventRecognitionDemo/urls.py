from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'VisualEventRecognitionDemo.views.home', name='home'),
    # url(r'^VisualEventRecognitionDemo/', include('VisualEventRecognitionDemo.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:

    url(r'^index/$', 'YoutubeVideos.views.index'),
    url(r'^mode1/$', 'YoutubeVideos.views.recognition'),
    url(r'^admin/', include(admin.site.urls)),

    # Mode one
    url(r'^previewTrainData/$', 'YoutubeVideos.views.previewTrainData'),
    url(r'^train/$', 'YoutubeVideos.views.trainSVM'),
    url(r'^previewTestData/$', 'YoutubeVideos.views.selectTestVideos'),
    url(r'^test/$', 'YoutubeVideos.views.testSVM'),

    # Mode two
    url(r'^mode2/$', 'ModeTwo.views.index'),
    url(r'^startFrames/$', 'ModeTwo.views.getFrames'),
    url(r'^frameProgressBar/$', 'ModeTwo.views.frameExtractProgress'),
    url(r'^startFeatures/$', 'ModeTwo.views.getFeatures'),
    url(r'^featureProgressBar/$', 'ModeTwo.views.featureExtractionProgress'),
    url(r'^histogram/$', 'ModeTwo.views.convertToHistogram'),
    url(r'^histogramProgressBar/$', 'ModeTwo.views.histogramProgressFun'),


    url(r'^previewTrainDataMode2/$', 'ModeTwo.views.previewTrainData'),
    url(r'^trainMode2/$', 'ModeTwo.views.trainSVM'),

    url(r'^classify/$', 'ModeTwo.views.classifyInputVideo'),
    url(r'^classifyProgress/$', 'ModeTwo.views.classifyProgress'),

)

from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns += staticfiles_urlpatterns()