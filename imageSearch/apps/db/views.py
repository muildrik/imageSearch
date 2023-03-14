import imghdr, pickle, multiprocessing
from io import BytesIO
from base64 import b64encode
from datetime import datetime
from .storage import DBO, FS, DeepLearningModel
from django.urls import reverse
from django.http import HttpResponseRedirect, Http404
from django.shortcuts import render
from django.contrib import messages
from bson.binary import Binary
from PIL import Image
from queue import Queue

BASE_DIR = 'imageSearch/media'

matches = []
matching_image = {}

dbo = DBO()
fs = FS()
dlm = DeepLearningModel()

def index(request):
    context = { 
        'msg' : False,
        'docs' : dbo.getAll('Projects'),
        'matching_files' : get_matches(),
        'matched_against' : get_matching_image()
    }
    return render(request, "db/index.html", context)

def all_projects(request):
    # context = render_to_string('db/projects.html', dbo.getAll('Projects'))
    if request.method == 'POST':
        return None
    else:
        return render(request, 'db/projects.html', dbo.getAll('Projects'))

def train_model(request, project_id):
    # queue = Queue()

    # TODO: Offload this to a worker to not block MainThread
    # TODO: Present updates to user with status of worker
    idfs, layer_normalizations, tensor_feature_vectors = dlm.train(project_id)
    
    update = {
        '$set' : {
            'trained' : True,
            'model' : {
                'idfs' : Binary(pickle.dumps(idfs, protocol=2), subtype=128),
                'layer_normalization' : Binary(pickle.dumps(layer_normalizations, protocol=2), subtype=128),
                'tensor_feature_vectors' : Binary(pickle.dumps(tensor_feature_vectors, protocol=2), subtype=128)
            }
        }
    }
    res = dbo.updateOne('Projects', project_id, update)
    if res:
        messages.success(request, 'Successfully trained model')
    else:
        messages.error(request, 'Failure training model')
    return HttpResponseRedirect(reverse('index'))

def analyze(request):
    if request.method == 'POST':
        if len(list(request.FILES.keys())) >= 1:
            project_id = list(request.FILES.keys())[0]
            files = request.FILES.getlist(project_id)
            
            # global matching_image
            
            matching_image, possible_matches = upload_file(project_id, files, True)
            
            if len(possible_matches) == 0:
                messages.success(request, f'No match found for this image in selected project dataset: analyzed {project_id}.')
            else:
                res = dbo.getByID('Projects', project_id)

                matching_files = []

                for match in list(possible_matches.keys()):
                    for file in res['files']:
                        if file['file_id'] == match:
                            matching_files.append({ 
                                'file_id' : file['file_id'],
                                'confidence' : possible_matches[match],
                                'name' : file['file_name'],
                                'base64' : file['b64url'] 
                            })
                
                global matches
                matches = sorted(matching_files, key=lambda d: d['confidence'], reverse=True)

        else:
            messages.warning(request, 'You did not select a file. Please select one or more files to analyze.')
        return HttpResponseRedirect(reverse('index'))
    else:
        return Http404()

def get_matches():
    global matches
    p = matches
    matches = []
    return p

def get_matching_image():
    global matching_image
    p = matching_image
    matching_image = {}
    return p

def new_project(request):
    if request.method == 'POST':
        data = { x[0] : x[1] for x in list(request.POST.items()) }
        if type(data) == dict:
            if 'csrfmiddlewaretoken' in data: del data['csrfmiddlewaretoken']
            res = dbo.insertOne('Projects', data)
            messages.success(request, 'Added a new project') if res else messages.error(request, 'Could not add a new project')
        else:
            messages.warning(request, 'You did not provide correct data')
        return HttpResponseRedirect('index')
    else:
        return render(request, 'db/new_project.html')

def new_image(request):
    if request.method == 'POST':
        if len(list(request.FILES.keys())) >= 1:
            project_id = list(request.FILES.keys())[0]
            files = request.FILES.getlist(project_id)
            # if len(files) == 1: files
            upload_file(project_id, files)
            messages.success(request, 'Added new images to your project. Please retrain the model of this project.')
        else:
            messages.warning(request, 'You did not select a file. Please select one or more files to add to the project.')
        return HttpResponseRedirect(reverse('index'))
    else:
        return Http404()

def del_image(request, project, image):
    if request.method == 'GET':
        del_file(project, image)
        messages.success(request, 'Image successfully removed from project')
        return HttpResponseRedirect(reverse('index'))

# def recording(request):
#     if request.method == 'POST':
#         form = ImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             upload_file(request.FILES['file'])
#             return HttpResponseRedirect('/success/url/')
#     else:
#         form = ImageForm()
#     return render(request, 'db/recording.html', { 'form' : form })

def empty_project(request, project_id:str):
    if request.method == 'GET':
        dir = f'{BASE_DIR}/{project_id}'
        res = dbo.delProp('Projects', project_id, 'files')
        dir = fs.rmdir(dir)
        if res:
            messages.success(request, 'Successfully removed the project\'s.')
        else:
            messages.error(request, 'There was a problem removing this project\'s data of associated files.')
        if dir:
            messages.success(request, 'Successfully removed all project files.')
        else:
            messages.error(request, 'There was a problem removing this project\'s files.')
        return HttpResponseRedirect(reverse('index'))
    else:
        return Http404()

def clean_project(request, project_id):
    if request.method == 'GET':
        dir = f'{BASE_DIR}/{project_id}'
        files_on_disk = fs.getDir(dir)
        res = dbo.getByID('Projects', project_id)

        msg = []

        if not files_on_disk or not res:
            messages.error(request, 'No files on disk or project does not exist')
            # DELETE PROJECT ID ON DISK IF ANY
            # DELETE PROJECT ID IN DATABASE IF ANY
        else:
            if res and 'files' in res:
                
                files_count, record_count = 0, 0
                removed_files, updated_records = [], []

                filenames_in_db = [x['file_id'] for x in res['files']]
                for file_name in files_on_disk:

                    # Remove dangling files on disk
                    if file_name not in filenames_in_db:
                        
                        files_count = files_count + 1
                        removed_files.append(file_name)
                        
                        fs.delfile(f'{dir}/{file_name}')
                        files_on_disk.remove(file_name)

                for file_name in filenames_in_db:
                    
                    if file_name not in files_on_disk:
                        record_count = record_count + 1
                        updated_records.append(file_name)

                        print(file_name)
                        # SET file_path TO SOME KIND OF DEFAULT
                    
                    # UPDATE DB RECORD

                if files_count > 0:
                    msg.append(f'Cleaned up the following {files_count} files: {", ".join(removed_files)}.')
                if record_count > 0: 
                    msg.append(f'The following {record_count} records had no associated file (paths have been set to default): {", ".join(updated_records)}.')
            else:

                # No files attribute on project: get rid of all stored dangling files for now
                res = fs.rmdir(dir)
                msg.append('Successfully removed all files associated with this project.')

                # TODO: ADD REINDEX OPTION SUCH THAT WHEN A DOCUMENT IS CORRUPTED NOT ALL FILES NEED TO BE UPLOADED AGAIN

        if len(msg) > 1:
            msg.pop(0)

        messages.success(request, ". ".join(msg))

        return HttpResponseRedirect(reverse('index'))

        # return render(request, 'db/index.html')
    else:
        return Http404()

def del_file(project_id, file_id):
    dir = f'{BASE_DIR}/{project_id}/{file_id}'
    metadata = { 'id' : file_id }
    update = { '$pull' : { 'files' : metadata }}
    res = dbo.updateOne('Projects', project_id, update)
    return True if type(res) is dict and fs.delfile(dir) else False

def upload_file(project_id:str, files, for_analysis:bool=False) -> bool:
    dir = f'{BASE_DIR}/{project_id}'
    file_stamp = datetime.timestamp(datetime.now())

    if fs.chkDir(dir, w=True):

        metadata = []
        file_paths = {}

        for filebuffer in files:
        
            file_name = f'{file_stamp}&^!%{filebuffer.name}'
            file_path = f'{dir}/{file_name}'

            file_paths[filebuffer.name] = file_path
            
            # IF FILE IS AN IMAGE
            if imghdr.what(filebuffer.file):
                buffer = thumbnail_image(filebuffer.file)
                b64 = f'data:{filebuffer.content_type};base64,{b64encode(buffer.getvalue()).decode("utf-8")}'
                new_file = { 'file_id' : file_name, 'file_name' : filebuffer.name, 'size' : filebuffer.size, 'content_type' : filebuffer.content_type, 'upload_time' : file_stamp, 'file_path' : file_path, 'b64url' : b64}
                metadata.append(new_file)

        field = 'files' if not for_analysis else 'analysis'
        
        update = { '$push' : { field : { '$each' : metadata } } }

        if not for_analysis:
            update['$set'] = { 'trained' : False }

        res = dbo.updateOne('Projects', project_id, update)
        
        if type(res) is dict:
            for filebuffer in files:
                with open(file_paths[filebuffer.name], 'wb+') as destination:
                    for chunk in filebuffer.chunks():
                        destination.write(chunk)

    if for_analysis:
        return new_file, dlm.image(project_id, file_name)
    else:
        return True

def thumbnail_image(file:BytesIO) -> BytesIO:
    thumb = Image.open(file)
    thumb.thumbnail((100, 100))

    # BUFFER FOR THE THUMBNAIL
    buf = BytesIO()
    
    # RESAVE THUMB TO BUFFER--WE DISCARD THUMB AFTERWARD BECAUSE WE'RE NOT SAVING THE ACTUAL THUMBNAIL TO DISC
    thumb.save(buf, format=thumb.format)
    
    return buf