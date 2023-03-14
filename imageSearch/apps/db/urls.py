from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('index', views.index, name='index'),
    path('new_image', views.new_image, name='new_image'),
    path('del_image/<str:project>/<str:image>', views.del_image, name='del_image'),
    path('analyze', views.analyze, name='analyze'),
    path('train_model/<str:project_id>', views.train_model, name='train_model'),
    # path('recording', views.recording, name='recording'),
    path('all_projects', views.all_projects, name='all_projects'),
    path('new_project', views.new_project, name='new_project'),
    path('clean_project/<str:project_id>', views.clean_project, name='clean_project'),
    path('empty_project/<str:project_id>', views.empty_project, name='empty_project'),
    # path('success', views.index, name='index'),
]