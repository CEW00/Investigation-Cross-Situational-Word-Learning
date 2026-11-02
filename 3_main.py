import os
import math
import pickle
import random
import itertools
from collections import namedtuple
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets, transforms
from argparse import ArgumentParser
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from types import SimpleNamespace
import torch

# set device so that needed data are accessible on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

"""
Script that conducts experiments with the created training and evaluation datasets.
    - Modified to work with new trial generation that saves all runs in ONE folder
    - Each run loads its specific experiment data (run_00.pt to run_19.pt)
"""

### Configure directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_images") 
RESIZED_DIR = os.path.join(BASE_DIR, "data", "resized_images") 
IM_RESIZED_DIR = os.path.join(BASE_DIR, "data", "resized_images", "images") 
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")


### experiment configuration 
EXP_TO_LOAD = 'ZF_ZF'  # name of experiment created in trial generation

EXP = 'hp_exp_two'                     # experiment: referential ambiguity: 'yu_smith'; zipfian: 'hp_exp_two'
PRES_MODE = 'scene_caption'            # presentation: several per scene: 'scene_caption'; one per scene: 'word_object'
DIS = 'zipfian'                        # distribution: 'uniform'; 'zipfian'

N_CAT = 28                             # total objects included: usually 18 for yu_smith, 28 for hp_exp_two
N_OBJ = 4                             # nr of objects per scene: usually 2,3,4 for yu_smith, 4 for hp_exp_two
N_OCC = 10                             # occurence of each word per experiment: usually 6 for yu_smith, 10 for hp_exp_two
N_FOIL = 27                            # number of false referents during testing: usually 3 for yu_smith, 27 for hp_exp_two

NORM = True                            # normalize embeddings option (during loss and forward)
LOSS = 'SGCL'                          # Loss function: Original:'triplet_loss', New:'SGCL'
ALPHA_TEMP = 1                         # temperature used for alpha factor in SGCL



Memory = namedtuple('Memory', ('feature', 'caption'))

class ReplayMemory():
    """
    Memory object to store observed features and captions
    """

    def __init__(self, capacity, hparams):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.hparams = hparams

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Memory(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, caption, feature=None):
        """Samples from the replay memory without replacement"""
        
        if self.hparams.presentation_mode == 'scene_caption':
            mismatch_found = False
            # sample a mismatch whose caption differs from the current caption
            while not mismatch_found:
                    sample = random.sample(self.memory, batch_size)
                    mismatch_caption = sample[0].caption
                    if mismatch_caption != caption:
                        mismatch_found = True
                        return sample
                             
                    
        elif self.hparams.presentation_mode == 'word_object':      
            # sample a mismatch whose caption differs from the current caption
            mismatch_found = False
            # sample a mismatch whose caption differs from the current caption
            while not mismatch_found:
                    sample = random.sample(self.memory, batch_size)
                    mismatch_caption = sample[0].caption
                    mismatch_feature = sample[0].feature
                    if mismatch_caption != caption and mismatch_feature != feature:
                        mismatch_found = True
                        return sample
    

    
    def __len__(self):
        return len(self.memory)


class ImageEmbeddingNet(nn.Module):
    """
    Image embedding subnetwork
    """

    def __init__(self, hidden_dim, embedding_dim):
        super(ImageEmbeddingNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(4096, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class WordEmbeddingNet(nn.Module):
    """
    Word embedding subnetwork
    """

    def __init__(self, input_dim, embedding_dim):
        super(WordEmbeddingNet, self).__init__()
        self.word_embed = nn.Embedding(input_dim, embedding_dim)

    def forward(self, x):
        return self.word_embed(x)

    def get_embedding(self, x):
        return self.forward(x)


class LossFunctions:
    def __init__(self, margin, alpha_temp, normalize, model=None, forward=None, image_emb_net=None):
        self.margin = margin
        self.alphatemp = alpha_temp
        self.model = model 
        self.forward = forward
        self.image_emb_net = image_emb_net
        self.normalize = normalize

        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device('cpu')
    

    def _contrastive_loss(self, match_embedding, mismatch_embedding):
        losses = F.relu(mismatch_embedding - match_embedding + self.margin).squeeze()

        if torch.isnan(losses):
            assert False

        return losses.mean()

    def VL_loss_scene_caption(self, match_attention_map, mismatch_caption_attention_map, mismatch_feature_attention_map, match_feature_n_objs):
        # calculate embedding based on max referent embedding per word
        # then average across all words
        match_ys = F.max_pool1d(match_attention_map, match_feature_n_objs)
        match_ys = match_ys.squeeze(2)
        match_embedding = torch.mean(match_ys, dim=1)

        mismatch_caption_ys = F.max_pool1d(mismatch_caption_attention_map, match_feature_n_objs)
        mismatch_caption_ys = mismatch_caption_ys.squeeze(2)
        mismatch_caption_embedding = torch.mean(mismatch_caption_ys, dim=1)

        mismatch_feature_ys = F.max_pool1d(mismatch_feature_attention_map, match_feature_n_objs)
        mismatch_feature_ys = mismatch_feature_ys.squeeze(2)
        mismatch_feature_embedding = torch.mean(mismatch_feature_ys, dim=1)

        # calculate triplet loss
        loss = (self._contrastive_loss(match_embedding, mismatch_caption_embedding) +
                self._contrastive_loss(match_embedding, mismatch_feature_embedding))
        
        return loss 

    def VL_loss_word_object(self, match_attention_map, mismatch_caption_attention_map, mismatch_feature_attention_map):
        match_embedding = match_attention_map.squeeze()
        mismatch_caption_embedding = mismatch_caption_attention_map.squeeze()
        mismatch_feature_embedding = mismatch_feature_attention_map.squeeze()

        # calculate triplet loss
        loss = (self._contrastive_loss(match_embedding, mismatch_caption_embedding) +
                self._contrastive_loss(match_embedding, mismatch_feature_embedding))
        
        return loss

    def alpha(self, feature_tensor, mismatch_feature):
        s_intra_value = self.s_intra(feature_tensor)
        s_cross_value = self.s_cross(feature_tensor, mismatch_feature)
        
        print(f"S_intra: {s_intra_value.item():.4f}, S_cross: {s_cross_value.item():.4f}")
        
        alpha = torch.exp(s_intra_value/self.alphatemp)/(torch.exp(s_intra_value/self.alphatemp) + torch.exp(s_cross_value/self.alphatemp))
                                                    
        print(f"Alpha: {alpha.item():.4f}")
        return alpha

    def s_intra(self, feature_tensor):
        xs = self.image_emb_net(feature_tensor.view(-1, 4096))
        
        # Add normalization to match your forward pass
        if self.model.hparams.normalized:
            xs = F.normalize(xs, p=2, dim=-1)
        
        xs = xs.view(feature_tensor.size(0), feature_tensor.size(1), -1)
        
        # Flatten to 2D: [batch_size * n_objects, embedding_dim]
        xs_flat = xs.view(-1, xs.size(-1))

        s_intra_full_matrix = torch.einsum('ik,jk->ij', xs_flat, xs_flat)
        N = s_intra_full_matrix.size(0)

        mask_diagnonal = torch.eye(N, device=s_intra_full_matrix.device).bool()
        s_intra_no_diag = s_intra_full_matrix[~mask_diagnonal]
        s_intra = torch.mean(s_intra_no_diag)/2

        return s_intra

    
    def s_cross(self, feature_tensor, mismatch_feature_tensor):
        xs_current = self.image_emb_net(feature_tensor.view(-1, 4096))
        xs_memory = self.image_emb_net(mismatch_feature_tensor.view(-1, 4096))
        
        # Add normalization to match your forward pass
        if self.model.hparams.normalized:
            xs_current = F.normalize(xs_current, p=2, dim=-1)
            xs_memory = F.normalize(xs_memory, p=2, dim=-1)
        
        xs_current = xs_current.view(feature_tensor.size(0), feature_tensor.size(1), -1)
        xs_current_flat = xs_current.view(-1, xs_current.size(-1))
        
        xs_memory = xs_memory.view(mismatch_feature_tensor.size(0), mismatch_feature_tensor.size(1), -1)
        xs_memory_flat = xs_memory.view(-1, xs_memory.size(-1))

        s_cross_full_matrix = torch.einsum('ik,jk->ij', xs_current_flat, xs_memory_flat)
        s_cross = torch.mean(s_cross_full_matrix)

        return s_cross
                
    def loss_null_word_object(self, feature_tensor, caption_tensor):
        feature_tensor = feature_tensor.to(self.device)
        caption_tensor = caption_tensor.to(self.device)

        binary_labels = torch.as_tensor([1], dtype=torch.float).to(self.device)

        # perform forward pass to get embeddings and attention maps
        xs, ws, attention_maps = self.forward(feature_tensor, caption_tensor)

        # since there is a single word-object, we can squeeze to get embedding directly
        outputs = torch.sigmoid(attention_maps).squeeze().unsqueeze(0)

        criterion = nn.BCELoss()
        loss = criterion(outputs, binary_labels)
        loss = 0 * loss

        return loss

    def loss_null_scene_caption(self, feature_tensor, caption_tensor):
        feature_tensor = feature_tensor.to(self.device)
        caption_tensor = caption_tensor.to(self.device)

        binary_labels = torch.as_tensor([1], dtype=torch.float).to(self.device)
        n_objs = feature_tensor.size(1)

        # perform forward pass to get embeddings and attention maps
        xs, ws, attention_maps = self.forward(feature_tensor, caption_tensor)

        # calculate output by taking max over objects for each word
        ys = F.max_pool1d(torch.sigmoid(attention_maps), n_objs)
        ys = ys.squeeze(2)
        outputs = torch.prod(ys, dim=1)

        criterion = nn.BCELoss()
        loss = criterion(outputs, binary_labels)
        loss = 0 * loss

        return loss
    

class ImageCaptionModel(pl.LightningModule):
    """
    Image caption model
    """

    def __init__(self, hparams, directory):
        super(ImageCaptionModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.img_size = 224
        self.vgg_embedding_dim = 4096
        self.hidden_dim = self.hparams.hidden_dim
        self.embedding_dim = self.hparams.embedding_dim
        self.image_embedding_net = self.init_image_embedding_net(
            self.hidden_dim, self.embedding_dim)
        self.word_embedding_net = self.init_word_embedding_net(
            self.hparams.n_categories, self.embedding_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.replay_memory = ReplayMemory(1000, hparams)
        self.base_dir = directory
        self.experiment_dir = os.path.join(self.base_dir, "experiments")
        
        # Load experiment data for the specific run
        self._load_experiment_data(self.experiment_dir)
        
        self.normalized = hparams.normalized
        self.val_results = []
        self.test_results = []
        self.test_outputs = []

        if self.hparams.save_embeddings:
            self.image_embeddings = []
            self.word_embeddings = []

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, 
                            momentum=self.hparams.momentum)

    def _load_experiment_data(self, experiment_dir):
        """Load pre-generated experiment data for specific run"""
        # Build path to the specific run file
        run_filename = f"run_{self.hparams.run_id:02d}.pt"
        experiment_path = os.path.join(experiment_dir, self.hparams.experiment_to_load, run_filename)
        
        print(f"Loading experiment data from: {experiment_path}")
        self.experiment_data = torch.load(experiment_path, weights_only=False)

        device = self.device

        # Override hparams with the ones from the saved experiment
        loaded_hparams = self.experiment_data['original_hparams']
        
        # Update the hparams properly
        for key, value in loaded_hparams.items():
            if hasattr(self.hparams, key):
                setattr(self.hparams, key, value)
        
        # Store the tensors that dataloaders will use
        self.train_features = self.experiment_data['train_features'].to(device)
        self.train_caption_ids = self.experiment_data['train_caption_ids'].to(device)
        
        # Load evaluation data for testing and plotting
        self.eval_features = self.experiment_data['eval_features'].to(device)
        self.eval_caption_ids = self.experiment_data['eval_caption_ids'].to(device)

        self.labels = self.experiment_data['labels']

    def train_dataloader(self):
        train_dataset = TensorDataset(self.train_features, self.train_caption_ids)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        """Use actual evaluation data for testing"""
        test_dataset = TensorDataset(self.eval_features, self.eval_caption_ids)
        return DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    def val_dataloader(self):
        """Validation dataloader - return None to skip validation"""
        return None

    def init_image_embedding_net(self, hidden_dim, embedding_dim):
        image_embedding_net = ImageEmbeddingNet(hidden_dim, embedding_dim)
        return image_embedding_net

    def init_word_embedding_net(self, n_categories, embedding_dim):
        word_embedding_net = WordEmbeddingNet(n_categories, embedding_dim)
        return word_embedding_net

    def forward(self, xs, ws):
        n_cells = xs.size(1)

        xs = xs.view(-1, 4096)
        xs = self.image_embedding_net(xs)
        xs = xs.view(-1, n_cells, self.embedding_dim)

        # perform word embedding
        ws = self.word_embedding_net(ws)

        if self.normalized:
            xs = F.normalize(xs, p=2, dim=-1)
            ws = F.normalize(ws, p=2, dim=-1)
    
            embed_dot_prod = torch.einsum('ikl,ijl->ijk', [xs, ws])
        else:
            embed_dot_prod = torch.einsum('ikl,ijl->ijk', [xs, ws])
            

        if self.hparams.sim == 'scaled_dot_prod_max':
            attention_map = torch.sigmoid(
                embed_dot_prod / math.sqrt(self.embedding_dim))
        elif self.hparams.sim == 'dot_prod_max' or self.hparams.sim == 'dot_prod_mean':
            attention_map = embed_dot_prod

        return xs, ws, attention_map

    def training_step(self, batch, batch_idx):
        feature, captions = batch

        feature = feature.to(self.device)  
        captions = captions.to(self.device)
        
        # Initialize loss functions with model reference
        loss_functions = LossFunctions(margin=self.hparams.margin, normalize=self.normalized, alpha_temp=self.hparams.alpha_temp, model=self, forward=self.forward, image_emb_net=self.image_embedding_net)

        ### TRIPLET LOSS AND WORD SCENE CAPTION   
        # sample mismatch indices from previous scenes
        match_feature_n_objs = feature.size(1)
        feature = feature[0].tolist()
        caption = captions[0].tolist()  
        
        # convert to torch tensor
        feature_tensor = torch.as_tensor(feature, dtype=torch.float).unsqueeze(0)
        caption_tensor = torch.as_tensor(caption, dtype=torch.long).unsqueeze(0)

        # handle first trial with no scene in memory
        n_mismatches = 1
        if len(self.replay_memory) >= n_mismatches:
            # Sampling mismatch is now handled in replay memory         
            mismatch_sample = self.replay_memory.sample(1, caption_tensor, feature_tensor if self.hparams.presentation_mode == 'word_object' else None)

            mismatch_feature = mismatch_sample[0].feature
            mismatch_caption = mismatch_sample[0].caption

            mismatch_feature_tensor = torch.as_tensor(mismatch_feature, dtype=torch.float).unsqueeze(0)
            mismatch_caption_tensor = torch.as_tensor(mismatch_caption, dtype=torch.long).unsqueeze(0)

            _, _, match_attention_map = self.forward(feature_tensor, caption_tensor)
            _, _, mismatch_caption_attention_map = self.forward(feature_tensor, mismatch_caption_tensor)
            _, _, mismatch_feature_attention_map = self.forward(mismatch_feature_tensor, caption_tensor)

            if self.hparams.presentation_mode == 'scene_caption':
                loss = loss_functions.VL_loss_scene_caption(match_attention_map, mismatch_caption_attention_map, mismatch_feature_attention_map, match_feature_n_objs)
                if self.hparams.loss_fn == 'triplet_loss':
                    loss = loss

                elif self.hparams.loss_fn == 'SGCL':
                    alpha_value =  loss_functions.alpha(feature_tensor, mismatch_feature_tensor)
                    loss = loss * alpha_value
                    
            elif self.hparams.presentation_mode == 'word_object':                  
                loss = loss_functions.VL_loss_word_object(match_attention_map, mismatch_caption_attention_map, mismatch_feature_attention_map)
        
        else:
            if self.hparams.presentation_mode == 'word_object':
                loss = loss_functions.loss_null_word_object(feature_tensor, caption_tensor)

            elif self.hparams.presentation_mode == 'scene_caption':
                loss = loss_functions.loss_null_scene_caption(feature_tensor, caption_tensor)
       
        # push current matching feature and caption to memory (only for online training, and not batch)
        if self.hparams.training_mode == 'online':
            if self.hparams.presentation_mode == 'scene_caption' or self.hparams.presentation_mode == 'word_object':
                self.replay_memory.push(feature, caption)

        # save embeddings here
        if self.hparams.save_embeddings:
            # This part needs the original dataset which we don't have access to anymore
            # You might need to reconstruct this part based on your evaluation data
            pass

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        features, captions = batch
        _, _, test_attention_maps = self.forward(features, captions)

        correct_idx = 0  # correct item is always first in the array
        test_accuracy = np.zeros(self.hparams.n_categories)

        # calculate accuracy
        if self.hparams.choice_fn == 'max' or self.hparams.choice_fn == 'softmax':
            # standard choice
            for i in range(len(test_attention_maps)):
                test_attention_map = test_attention_maps[i].flatten()
                max_attention_map_idx = torch.argmax(test_attention_map)
                if max_attention_map_idx == correct_idx:
                    test_accuracy[i] = 1

        if self.hparams.experiment.startswith('yu_smith'):
            accuracy = np.array(np.sum(test_accuracy) / self.hparams.n_categories)
            result = {'test_accuracy': accuracy}
        elif self.hparams.experiment.startswith('hp_exp_two'):
            if self.hparams.dist == 'uniform':
                matched_accuracy = np.array(np.sum(test_accuracy) / self.hparams.n_categories)
                result = {'high_accuracy': None, 'matched_accuracy': matched_accuracy, 'low_accuracy': None}
            elif self.hparams.dist == 'zipfian':
                high_accuracy = np.array(np.sum(test_accuracy[0:4]) / 4)
                matched_accuracy = np.array(np.sum(test_accuracy[4:16]) / 12)
                low_accuracy = np.array(np.sum(test_accuracy[16:28]) / 12)
                result = {'high_accuracy': high_accuracy, 'matched_accuracy': matched_accuracy, 'low_accuracy': low_accuracy}
            else:
                result = {}

        self.test_outputs.append(result)
        return result
    
    def on_test_epoch_end(self):
        # Store outputs from test_step in instance attributes
        if hasattr(self, 'test_outputs'):
            outputs = self.test_outputs
            
            hparam_vals = []
            for hparam in self.hparams:
                hparam_vals.append(getattr(self.hparams, hparam))

            if self.hparams.experiment.startswith('yu_smith'):
                test_accuracy = np.array([output['test_accuracy']
                                        for output in outputs])
                self.test_results = hparam_vals + \
                    [float(test_accuracy[0])]
            elif self.hparams.experiment.startswith('hp_exp_two'):
                high_accuracy = np.array([output['high_accuracy'] for
                                        output in outputs])
                matched_accuracy = np.array([output['matched_accuracy'] for
                                            output in outputs])
                low_accuracy = np.array([output['low_accuracy'] for
                                        output in outputs])
                self.test_results = hparam_vals + [(high_accuracy[0]),
                                                float(matched_accuracy[0]),
                                                (low_accuracy[0])]
            # Clear the stored outputs
            self.test_outputs = []

    def generate_plot_data(self):
        """
        Generates scenes containing all images and a single caption
        Now uses the actual evaluation data structure
        """
        # Use the evaluation data that's already properly structured
        plot_scenes = np.zeros((self.hparams.n_categories, self.hparams.n_foils + 1,
                                self.img_size, self.img_size, 3))
        plot_features = self.eval_features.cpu().numpy()
        plot_caption_ids = self.eval_caption_ids.cpu().numpy()
        plot_captions = [self.labels[id_] for id_ in range(self.hparams.n_categories)]

        return plot_scenes, plot_features, plot_captions, plot_caption_ids

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # experiment arguments
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--experiment_to_load', type=str, required=False, default=EXP_TO_LOAD,
                        help='name of pre-generated experiment to load')
        parser.add_argument('--experiment', type=str, required=False, default=EXP,
                            help='experiment name and directory to save results to')
        parser.add_argument('--run_id', type=int, default=0,
                            help='which run to load (0-19)')

        # model arguments
        parser.add_argument('--hidden_dim', type=int, required=False, default=400, 
                            help='number of hidden units for the vision embedding')
        parser.add_argument('--embedding_dim', type=int, required=False, default=64,
                            help='dimensionality of vision and word embedding')
        parser.add_argument('--learning_rate', type=float, required=False, default=0.01,
                            help='learning rate')
        parser.add_argument('--momentum', type=float, required=False, default=0.0,
                            help='momentum for SGD')
        parser.add_argument('--batch_size', type=int, required=False, default=1,
                            help='batch size')
        parser.add_argument('--training_mode', type=str, choices=['online', 'batch'], default='online',
                            required=False, help='set training mode to be online (one example using memory) or batch style training')
        parser.add_argument('--presentation_mode', type=str, default=PRES_MODE,
                            choices=['scene_caption', 'word_object'], required=False, 
                            help='set presentations to use scene-caption pairs or word-object pairs')

        parser.add_argument('--features', type=str, choices=['pretrained'], default='pretrained',
                            help='choice of random or pretrained features for cnn')
        parser.add_argument('--sim', type=str, choices=['dot_prod_max', 'scaled_dot_prod_max'], required=False, default='dot_prod_max',
                            help='choice of scaled or raw dot product for calculating similarity')
        parser.add_argument('--grad_clip_val', type=float, default=10.0,
                            help='amount of gradient clipping to apply')

        parser.add_argument('--normalized', action='store_true', default=NORM,
                    help='Use normalized embeddings in forward pass')
        
        parser.add_argument('--loss_fn', type=str, default=LOSS,
                    choices=['binary_cross_entropy', 'triplet_loss', 'SGCL'], required=False,
                    help='choice of loss function (bce or triplet loss)')

        # triplet loss arguments
        parser.add_argument('--triplet_loss_type', type=str,
                            choices=['joint'], default='joint',
                            help='choice of loss type when using triplet loss')
        parser.add_argument('--margin', default=1, type=float,
                            help='margin for triplet loss')
        parser.add_argument('--alpha_temp', default=ALPHA_TEMP, type=float,
                            help='temperature for the alpha scalar calculation')
        
        # experiment arguments
        parser.add_argument('--n_categories', type=int, required=False, default=N_CAT,
                            help='total number of categories')
        parser.add_argument('--n_objs', type=int, required=False, default=N_OBJ,
                            help='number of objects per scene')
        parser.add_argument('--n_occurrences', type=int, default=N_OCC,
                            help='number of occurrences of each category throughout the task')
        parser.add_argument('--n_foils', type=int, required=False, default=N_FOIL,
                            help='number of foils during evaluation')
        parser.add_argument('--n_mismatches', required=False, default=1,
                            help='number of mismatches to sample during learning')
        parser.add_argument('--choice_fn', type=str, choices=['max'],
                            required=False, default='max',
                            help='choice function (softmax or max)')
        parser.add_argument('--temp', default=1/100, type=float,
                            help='temperature of softmax')
        parser.add_argument('--save_results', action='store_true', default=True,
                            help='save results to directory')
        parser.add_argument('--save_embeddings', action='store_true',
                            help='save embeddings after each trial')
        parser.add_argument('--plot_results', action='store_true', default=True,
                            help='plot results')

        # hendrickson and perfors (2019) arguments
        parser.add_argument('--dist', type=str, choices=['uniform', 'zipfian'], default=DIS,
                            help='distribution to use for HP experiments (uniform or zipfian)')

        return parser


def main(hparams, run_id):
    # Set the run_id in hparams so the model can load the correct data
    hparams.run_id = run_id

    # init module
    model = ImageCaptionModel(hparams, BASE_DIR)

    # use trainer with some modified settings
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator='gpu' if hparams.gpus else 'cpu',
        devices=hparams.gpus,
        num_nodes=hparams.nodes,
        check_val_every_n_epoch=1,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        gradient_clip_val=hparams.grad_clip_val
    )

    # train model
    trainer.fit(model)

    # evaluate model
    trainer.test(ckpt_path=None)
    results = model.test_results

    # create results directory once
    res_dir = os.path.join(RESULTS_DIR, hparams.experiment)
    os.makedirs(res_dir, exist_ok=True)

    # save model and results
    if hparams.save_results:
        checkpoint_path = os.path.join(res_dir, f'run_{run_id:02d}.ckpt')
        trainer.save_checkpoint(checkpoint_path)

    # save embeddings
    if hparams.save_embeddings:
        # extract saved embeddings
        image_embeddings = np.array(model.image_embeddings)
        word_embeddings = np.array(model.word_embeddings)

        # save embeddings
        image_embedding_path = os.path.join(res_dir, f'{hparams.n_objs}_objs_image_embeddings.npy')
        word_embedding_path = os.path.join(res_dir, f'{hparams.n_objs}_objs_word_embeddings.npy')
        
        with open(image_embedding_path, 'wb') as f:
            np.save(f, image_embeddings)
        with open(word_embedding_path, 'wb') as f:
            np.save(f, word_embeddings)

    # plot attention
    if hparams.plot_results:
        plot_data = model.generate_plot_data()  
        plot_scenes, plot_scene_features, plot_captions, plot_caption_ids = plot_data
        plot_path = os.path.join(res_dir, f'{hparams.n_objs}_objs_attention_map.png')
        if hparams.experiment.startswith('yu_smith'):
            plot_path = os.path.join(res_dir, f'{hparams.n_objs}_objs_attention_map.png')
        elif hparams.experiment.startswith('hp_exp_two'):
            plot_path = os.path.join(res_dir, f'dist_{hparams.dist}_run_{run_id}_attention_map.png')
        
        plt.savefig(plot_path, bbox_inches='tight')

    return results


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--n_runs', type=int, required=False, default=20)
    parser.add_argument('--max_epochs', type=int, required=False, default=1,
                        help='number of epochs to train for')
    parser.add_argument('--seed', type=int, required=False, default=1, 
                        help='seed')

    # give the module a chance to add own params
    parser = ImageCaptionModel.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    # set seed
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    # list to store results
    results = []

    # run simulations and append results - NOW USING DIFFERENT DATA FOR EACH RUN
    for run_id in range(hparams.n_runs):
        print(f"\n=== Starting Run {run_id+1}/{hparams.n_runs} ===")
        curr_results = main(hparams, run_id)
        results.append(curr_results)
        print(f"=== Completed Run {run_id+1}/{hparams.n_runs} ===")

    # extract hparams to store results in later
    hparam_cols = list(vars(hparams).keys())
    hparam_vals = [getattr(hparams, hparam) for hparam in hparam_cols]

    # add test accuracy column
    if hparams.experiment.startswith('yu_smith'):
        hparam_cols.append('test_accuracy')
    elif hparams.experiment.startswith('hp_exp_two'):
        hparam_cols.extend(['high_accuracy', 'matched_accuracy', 'low_accuracy'])
   
    if hparams.save_results:
        csv_path = os.path.join(RESULTS_DIR, hparams.experiment, f'{hparams.presentation_mode}.csv')
        df = pd.DataFrame(np.array(results), columns=hparam_cols)
        with open(csv_path, 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
    
    print(f"\n All {hparams.n_runs} runs completed successfully!")
    print(f" Results saved to: {csv_path}")