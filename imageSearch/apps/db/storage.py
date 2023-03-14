import os, torch, torchvision
import numpy as np
from cv2 import VideoCapture

import shutil

from torchvision import models, transforms, datasets

from torch.utils.data import DataLoader

from PIL import Image

import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist

from pymongo import MongoClient
from bson import ObjectId
from bson.binary import Binary
import pickle

import sys, io, random, requests

TRANSFORM = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

class DBO:

    def __init__(self):
        self.dbo = MongoClient()['imageSearch']

    def getCollections(self):
        return self.dbo.list_collections()

    def getCollection(self, name:str) -> str:
        return self.dbo.get_collection(name)

    def getAll(self, collection:str):
        docs = [x for x in self.dbo[collection].find(limit=20)]
        for doc in docs:
            id = doc['_id']
            doc['id'] = id
            del doc['_id']
        return { 
            'docs' : docs,
            'total' : self.dbo[collection].count_documents({})
        }

    def getByID(self, collection:str, object_id:str):
        return self.dbo[collection].find_one({ '_id' : ObjectId(object_id) })

    def delByID(self, collection:str, object_id:str):
        return self.dbo[collection].delete_one({ '_id' : ObjectId(object_id) })

    def delProp(self, collection:str, object_id:str, prop:str):
        return self.dbo[collection].find_one_and_update({ '_id' : ObjectId(object_id) }, { '$unset' : { prop : 1 }, '$set' : { 'trained' : False } }, return_document=False)

    def insertOne(self, collection:str, data:dict):
        return self.dbo[collection].insert_one(data)

    def updateOne(self, collection:str, id:str, update:dict, return_new:bool=True):
        return self.dbo[collection].find_one_and_update({ '_id' : ObjectId(id) }, update, return_document=return_new)
    
class FS:

    def __init__(self):
        self.bla = None

    def mkdir(self, path:str):
        os.makedirs(path, exist_ok=True)

    def chkDir(self, path:str, r:bool=False, w:bool=False, m:bool=True):
        if not os.path.exists(path) and m: self.mkdir(path)
        if r: return os.access(path, os.R_OK)
        if w: return os.access(path, os.X_OK | os.W_OK)
        return False

    def chkFile(self, path:str):
        if self.chkDir(path, r=True, m=False):
            return os.path.isfile(path)
    
    def getDir(self, path:str):
        if self.chkDir(path, r=True):
            return [x for x in os.walk(path)][0][2]

    def delfile(self, path:str) -> bool:
        if os.path.isfile(path) and os.access(path, os.X_OK | os.W_OK):
            os.remove(path)
            return True
        return False

    def rmdir(self, path:str) -> bool:
        if os.path.isdir(path) and os.access(path, os.X_OK | os.W_OK):
            shutil.rmtree(path, ignore_errors=True)
            return True
        return False

    def save(self, path:str, file:bytes):
        print('ok')

class RetrievalDatasetFromDirectory(torch.utils.data.Dataset):

    def __init__(self, image_root_dir:str, transform=None):
        self.image_paths = self.get_all_img_paths(image_root_dir)
        self.transform = transform
        
    def get_all_img_paths(self, img_dir:str) -> list:
        img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if(f[0] != "." and (f[-4:] == ".jpg" or f[-4:] == ".png")):
                    image_path = root + os.sep + f
                    img_paths.append(image_path)
        return img_paths
    
    def get_id_at_index(self, idx:int) -> str:
        return self.image_paths[idx]
        
    def get_image_at_index(self, idx:int) -> Image:
        return Image.open(self.image_paths[idx]).convert('RGB')
    
    def __getitem__(self, idx:int):
        x = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: x = self.transform(x)
        return x, idx
    
    def __len__(self) -> int:
        return len(self.image_paths)

class DeepLearningModel():

    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        self.base_dir = 'imageSearch/media'
        self.n_layers = 3
        self.dbo = DBO()

    def video(self, project_id:str, video_id:str) -> dict:
        file, idfs, layer_normalizations, dataset_tf_vecs = self.__load_data_from_db(project_id, video_id)
        video_frames = self.__read_in_video_as_tensor(file['file_path'])
        model = FullImgVGG(self.n_layers)
        framewise_activations = get_activations_in_batch_as_list(model, video_frames)
        framewise_tfs = self.__get_tf_vectors(layer_normalizations, framewise_activations)
        return self.__find_match(dataset_tf_vecs, framewise_tfs, idfs)

    def image(self, project_id:str, image_id:str) -> dict:
        file, idfs, layer_normalizations, dataset_tf_vecs = self.__load_data_from_db(project_id, image_id)
        image_as_tensor = self.__read_in_image_as_tensor(file['file_path'])
        model = FullImgVGG(self.n_layers)
        image_tfs = self.__get_tf_vec_for_image(model, image_as_tensor, layer_normalizations)
        return self.__find_match(dataset_tf_vecs, image_tfs, idfs)

    def train(self, project_id:str, finetune:bool=False):
        """
        Train the model with the methods below. Can be finetuned, but will take a moment to do (defaults to false).
        """
        dataset = self.__load_data_from_disk(project_id)
        
        img_ids = [y for x in dataset.image_paths for y in x.split('\\')[-1:]]

        dataloader = DataLoader(dataset, batch_size=7, shuffle=False, num_workers=2)
        
        model = FullImgVGG(self.n_layers)
        
        if finetune:
            self.__finetune_model(model, 5, dataloader, max(4, int(len(dataset) / 50)))
        
        idfs, layer_normalizations, tf_vecs = self.__calculate_feature_vectors(model, dataloader)

        tensor_feature_vectors = {img_id : tf_vecs[idx] for idx, img_id in enumerate(img_ids)}
        
        return idfs, layer_normalizations, tensor_feature_vectors

    def __load_data_from_disk(self, project_id:str):
        file_path = os.path.join(self.base_dir, project_id)
        return RetrievalDatasetFromDirectory(file_path, self.transform)
    
    def __load_data_from_db(self, project_id:str, file_id:str):
        res = self.dbo.getByID('Projects', project_id)
        if res:
            dataset_tf_vecs = pickle.loads(res['model']['tensor_feature_vectors'])
            idfs = pickle.loads(res['model']['idfs'])
            layer_normalizations = pickle.loads(res['model']['layer_normalization'])
            file = [file for file in res['analysis'] if file['file_id'] == file_id][0]
            return file, idfs, layer_normalizations, dataset_tf_vecs

    def __read_in_image_as_tensor(self, file_path:str):
        return self.transform(Image.open(file_path).convert('RGB')).unsqueeze(0)

    def __read_in_video_as_tensor(self, file_path:str):
        frames = []
        skip_frames = 10
        frame_count = 0
        vidcap = VideoCapture(file_path)
        success, frame = vidcap.read()
        while success:
            if (frame_count % skip_frames == 0):
                im_tensor = self.transform(Image.fromarray(frame).convert('RGB'))
                frames.append(im_tensor)
            success, frame = vidcap.read()
            frame_count += 1
        vid_as_tensor = torch.stack(frames)
        vidcap.release()
        return vid_as_tensor

    def __get_tf_vec_for_image(self, model, image_as_tensor, layer_normalizations):
        kernel_max_activations = model(image_as_tensor)
        kernel_max_activations = np.array([kern.squeeze(0).detach().numpy() for kern in kernel_max_activations], dtype=object)
        normalized_layer_rep = kernel_max_activations/np.array(layer_normalizations)    
        return np.expand_dims(np.concatenate(normalized_layer_rep), 0)

    def __get_tf_vectors(self, layer_normalizations, layer_representations):
        n_layers = len(layer_normalizations)
        max_activations=[]
        for l in range(n_layers):
            norm = layer_normalizations[l]
            layer = layer_representations[l]
            norm_layer_representations = layer/norm
            max_activations.append(norm_layer_representations)
        tf_vectors = np.concatenate(max_activations,axis=1)
        return tf_vectors

    def __find_match(self, dataset_tf_vecs, query_tfs, idfs):

        def find_most_confident_matches(dataset_tf_vecs, query_tfs, idfs):
            dataset_tf_array, index_to_id_dict = parse_list_from_dict(dataset_tf_vecs)
            dataset_tf_array = np.array(dataset_tf_array, dtype=np.float32)
            dataset_tf_idfs = dataset_tf_array * idfs
            query_tf_idfs = query_tfs * idfs

            def get_closest_imgs_in_db(db_tf_idf, query_tf_idf, num_retrieve=3):
                distances = cdist(query_tf_idf, db_tf_idf, metric='cosine')
                #cut down to only look at the 10 closest frames
                num_frames_to_observe = min(12, len(distances))
                closest_frames = np.argsort(np.sort(distances, axis=1)[:,0])[:num_frames_to_observe]
                closest_indices = np.argsort(distances,axis=1)[closest_frames][:,:num_retrieve]
                closest_distances = np.sort(distances,axis=1)[closest_frames][:,:num_retrieve]
                #print(closest_distances)
                return closest_indices, closest_distances
            
            closest_indices, closest_distances = get_closest_imgs_in_db(dataset_tf_idfs, query_tf_idfs, 3)

            def perfect_matches(idx_id_dict, closest_indices, closest_distances):
                exact_match_coords = np.nonzero(closest_distances <= 0.1)
                if exact_match_coords:
                    indices = closest_indices[exact_match_coords]
                else:
                    return None
                
                id_to_confidence = {}
                
                for i in indices:
                    img_id = idx_id_dict[i]
                    id_to_confidence[img_id] = 1
                
                return id_to_confidence

            def get_confidences_by_index(idx_id_dict, closest_indices, closest_distances):
                max_possible_weight = 5 * len(closest_indices)
                renormalized_distances = (closest_distances-0.1)/0.25
                inverse_distances = 1/renormalized_distances
                id_to_confidence = {}
                for idx in np.unique(closest_indices):
                    count_total_for_idx = np.sum(inverse_distances[np.nonzero(closest_indices==idx)])
                    confidence_for_idx = min(count_total_for_idx/max_possible_weight,0.95)
                    idx_id = idx_id_dict[idx]
                    id_to_confidence[idx_id]=confidence_for_idx
                return id_to_confidence

            # No good matches if distance is too great
            if np.all(closest_distances > 0.40): return {}

            # Perfect matches
            matches = perfect_matches(index_to_id_dict, closest_indices, closest_distances)
            if matches: return matches
            
            # Imperfect matches
            return get_confidences_by_index(index_to_id_dict, closest_indices, closest_distances)
        
        return find_most_confident_matches(dataset_tf_vecs, query_tfs, idfs)

    def __calculate_feature_vectors(self, model, dataloader):
        
        dataset_raw_model_outputs = get_layerwise_representations_of_all_images(model, dataloader)

        def find_layer_normalizations(layer_representations):
            max_neuron_activations = []
            for layer in layer_representations:
                max_neuron = np.max(layer)
                max_neuron_activations.append(float(max_neuron))
            return max_neuron_activations
    
        layer_normalizations = find_layer_normalizations(dataset_raw_model_outputs)

        def get_tf_vectors(layer_normalizations, layer_representations):
            n_layers = len(layer_normalizations)
            max_activations=[]
            for l in range(n_layers):
                norm = layer_normalizations[l]
                layer = layer_representations[l]
                norm_layer_representations = layer/norm
                max_activations.append(norm_layer_representations)
            tf_vectors = np.concatenate(max_activations, axis=1)
            return tf_vectors

        dataset_tf_vecs = get_tf_vectors(layer_normalizations, dataset_raw_model_outputs)

        def calculate_idf_by_summed_activations(all_tf_vectors):
            num_images = len(all_tf_vectors)
            sum_counts_per_filter = np.sum(all_tf_vectors,axis=0)
            inverse_percents = num_images/(sum_counts_per_filter+0.1)
            idf = np.log(inverse_percents)
            return idf

        idfs = calculate_idf_by_summed_activations(dataset_tf_vecs)

        return idfs, layer_normalizations, dataset_tf_vecs

    def __finetune_model(self, model, n_epochs, dataset_loader, num_nearby_images_to_avg_over):
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        loss_func = nn.CosineEmbeddingLoss()
        
        def run_epoch(model,dataset_loader,loss_func,optimizer,nearby_images=6):
    
            model.eval()
            entire_dataset_vecs = get_layerwise_representations_of_all_images(model,dataset_loader)
            model.train()
            total_loss = [0 for i in range(model.n_layers)]
            for b,_ in dataset_loader:
                batch_vecs = model(b)
                nearby_avg = get_avg_vec_of_closest(batch_vecs,entire_dataset_vecs,2)
                loss_by_layer = [loss_func(batch_vecs[i],torch.FloatTensor(nearby_avg[i]),torch.LongTensor([1])) for i in range(model.n_layers)]
                total_loss=[sum(x) for x in zip(total_loss,[l.item() for l in loss_by_layer])]
                torch.autograd.backward(loss_by_layer)
                optimizer.step()
            print(f"Loss for Epoch: {total_loss}")

            def get_avg_vec_of_closest(batch_tensors_by_layer,dataset_representation_by_layer,k):
                avg_closest_by_layer = []
                for layer in range(model.n_layers):
                    #still batched
                    vec = batch_tensors_by_layer[layer].detach().numpy()
                    layer_rep_across_db = dataset_representation_by_layer[layer]
                    layer_distances = cdist(vec,layer_rep_across_db,metric='cosine')
                    #print(layer_distances.shape)
                    closest_vec_indices_batched = np.argsort(layer_distances,axis=1)[:,:k]
                    closest_vecs_batched = layer_rep_across_db[closest_vec_indices_batched]
                    #print(closest_vecs_batched.shape)
                    average_closest_vec_batched = np.mean(closest_vecs_batched,axis=1)
                    avg_closest_by_layer.append(average_closest_vec_batched)
                return avg_closest_by_layer
        
        for i in range(n_epochs):
            print(f"Finetuning Epoch {i}+1...")
            run_epoch(model,dataset_loader,loss_func,optimizer,nearby_images=num_nearby_images_to_avg_over)

class RetrievalDatasetFromDirectory(torch.utils.data.Dataset):
    
    def __init__(self, image_root_dir:str, transform=None):
        self.image_paths = self.get_all_img_paths(image_root_dir)
        self.transform = transform
        
    def get_all_img_paths(self, img_dir:str) -> list:
        img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if(f[0] != "." and (f[-4:] == ".jpg" or f[-4:] == ".png")):
                    image_path = root + os.sep + f
                    img_paths.append(image_path)
        return img_paths
    
    def get_id_at_index(self, idx:int) -> str:
        return self.image_paths[idx]
        
    def get_image_at_index(self, idx:int) -> Image:
        return Image.open(self.image_paths[idx]).convert('RGB')
    
    def __getitem__(self, idx:int):
        x = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: x = self.transform(x)
        return x, idx
    
    def __len__(self) -> int:
        return len(self.image_paths)

class FullImgVGG(nn.Module):

    def __init__(self, n_layers):
        super(FullImgVGG, self).__init__()
        self.features = torchvision.models.vgg16().features
        # self.features = self.features.cuda()
        self.n_layers = n_layers
        if(n_layers == 3):
            self.groups = [self.features[0:16], self.features[16:23], self.features[23:30]]
        elif(n_layers == 4):
            self.groups = [self.features[0:9], self.features[9:16], self.features[16:23], self.features[23:30]]
        else:
            raise Exception("Only support using 3 or 4 convolutional activation maps")

    def forward(self, x):
        layerwise_representations = []
        prev_layer_act = x
        for group in self.groups:
            layer_act = group(prev_layer_act)
            layer_rep = layer_act.max(dim=2)[0].max(dim=2)[0]
            layerwise_representations.append(layer_rep)
            prev_layer_act = layer_act
        return layerwise_representations
    
def get_layerwise_representations_of_all_images(model, dataset_loader):
    representations_per_layer = [[] for i in range(model.n_layers)]

    for batch_ims, indices in dataset_loader:
        list_of_nump_arrays = get_activations_in_batch_as_list(model, batch_ims)
        
        #because each layer is a different size need to do some concatenation gymnastics
        for layer in range(model.n_layers):
            representations_per_layer[layer].append(list_of_nump_arrays[layer])
        # [representations_per_layer[i].append(list_of_nump_arrays[i]) for i in range(model.n_layers)]

    if len(representations_per_layer):
        representations_per_layer = [np.concatenate(lay) for lay in representations_per_layer]
    return representations_per_layer

def get_activations_in_batch_as_list(model, batch_images):
        print("I'm alive")
        with torch.no_grad():
            batch_representations = model(batch_images)
        batch_reps_as_list = [layer_rep.detach().numpy() for layer_rep in batch_representations]
        return batch_reps_as_list

def parse_list_from_dict(id_to_item_dict:dict):
    curr_index = 0
    item_list = []
    index_id_map = {}
    for im_id, item in id_to_item_dict.items():
        item_list.append(item)
        index_id_map[curr_index] = im_id
        curr_index += 1
    return item_list, index_id_map