from django.urls import path
from . import views

urlpatterns = [
    path('', views.review_view, name='review_view'),
]