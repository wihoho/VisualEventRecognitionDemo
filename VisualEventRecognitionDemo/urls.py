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
    url(r'^recognition/$', 'YoutubeVideos.views.recognition'),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^previewTrainData/$', 'YoutubeVideos.views.previewTrainData'),
    url(r'^train/$', 'YoutubeVideos.views.trainSVM'),
    url(r'^previewTestData/$', 'YoutubeVideos.views.selectTestVideos'),
    url(r'^test/$', 'YoutubeVideos.views.testSVM'),




)

from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns += staticfiles_urlpatterns()