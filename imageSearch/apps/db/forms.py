from .models import Project, Image, Recording
from django import forms
 
class ProjectForm(forms.ModelForm):
    # title = forms.CharField()
    # description = forms.CharField()
    # file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
    class Meta:
        model = Project
        fields=('title','description')

class ImageForm(forms.ModelForm):
    title = forms.CharField()
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
    
    class Meta:
        model = Project
        fields=('title','description')
    # description = forms.CharField()

class RecordingForm(forms.Form):
    title = forms.CharField()
    description = forms.CharField()
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))