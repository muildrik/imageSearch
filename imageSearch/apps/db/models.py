import json
from djongo import models
from pymongo import MongoClient

class Project(models.Model):
    _id = models.ObjectIdField()
    title = models.CharField(max_length=100)
    description = models.TextField()

    # def __str__(self):
        # return self.title

    # def save(self, *args, **kwargs):


        # return MongoClient()['imageSearch']['Projects'].insert_one(self)

    def delete(self):
        return MongoClient()['imageSearch']['Projects'].delete_one(self)

class Image(models.Model):
    project = models.EmbeddedField(model_container=Project,)
    headline = models.CharField(max_length=255)
    file_field = models.FileField(upload_to='db')
    objects = models.DjongoManager()

    class Meta:
        abstract = True

class Recording(models.Model):
    project = models.EmbeddedField(model_container=Project,)
    headline = models.CharField(max_length=255)
    file_field = models.FileField(upload_to='db')
    objects = models.DjongoManager()

    class Meta:
        abstract = True
  
