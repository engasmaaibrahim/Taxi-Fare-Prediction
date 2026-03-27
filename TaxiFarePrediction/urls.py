from django.urls import path
from .views import predict_view, index  

urlpatterns = [
    path("", index, name="home"), 
    path("api/predict/", predict_view, name="predict"),
]
