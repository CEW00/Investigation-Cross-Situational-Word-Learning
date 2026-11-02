import time
import os
import pickle
import random
import itertools
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from types import SimpleNamespace

"""
Script that generates experiments with training and evaluation datasets.
    - Generates 20 independent runs with different randomizations
    - Saves all runs in ONE folder for easy access
"""

### Configure directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "data")
RESIZED_DIR = os.path.join(DATA_DIR, "resized_images") 

### experiment configuration 
EXP_NAME ='ZF_ZF'  # save name of experiment to be specified in main

EXP = 'hp_exp_two'        # experiment: referential ambiguity: 'yu_smith'; zipfian: 'hp_exp_two'
PRES = 'scene_caption'  # presentation: several per scene: 'scene_caption'; one per scene: 'word_object'
DIST = 'zipfian'        # distribution: 'uniform'; 'zipfian'

N_CAT = 28              # total objects to be included 28
N_OBJ = 4             # nr of objects per scene
N_OCC = 10               # occurence of each word per experiment 10
N_FOILS = 27            # number of false referents during testing
N_RUNS = 20             # number of independent runs to generate
SEED = 42               # base seed for reproducibility


class NounDataset(Dataset):
    """
    Dataset class for creating experiments
    """

    def __init__(self, hparams, directory, run_id=0):
        # experiment variables
        self.base_dir = directory
        self.experiment_dir = os.path.join(self.base_dir, "experiments")            
        self.hparams = hparams
        self.run_id = run_id

        # Set different random seed for each run
        self.rng = random.Random(SEED + run_id)
        self.np_rng = np.random.default_rng(SEED + run_id)
        
        self.experiment = self.hparams.experiment

        if self.experiment.startswith('yu_smith'):
            self.n_categories = self.hparams.n_categories
            self.n_objs = self.hparams.n_objs
            self.n_occurrences = self.hparams.n_occurrences
            self.scene_ids = np.arange(
                self.n_categories).repeat(self.n_occurrences)
            # generate cooccurence ids
            self.scene_ids = self._generate_unique(
                self.scene_ids, self.hparams.n_objs)
            self.n_matching_scenes = len(self.scene_ids)
            self.n_foils = self.hparams.n_foils
        elif hparams.experiment.startswith('hp_exp_two'):
            self.n_categories = self.hparams.n_categories
            self.n_objs = self.hparams.n_objs
            self.n_occurrences = self.hparams.n_occurrences
            self.n_foils = self.hparams.n_foils

            if self.hparams.dist == 'uniform':
                # generate uniform cooccurence ids
                self.scene_ids = np.arange(self.n_categories).repeat(
                    self.n_occurrences)
                self.scene_ids = self._generate_unique(self.scene_ids,
                                                       self.hparams.n_objs)
                self.n_matching_scenes = len(self.scene_ids)
            elif self.hparams.dist == 'zipfian':
                # generate zipfian cooccurence ids
                zipfian_counts = [62, 33, 21, 19, 11, 11, 11, 11, 9, 9,
                                  8, 8, 8, 8, 8, 8, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1]
                self.scene_ids = []
                for i in range(self.n_categories):
                    for j in range(zipfian_counts[i]):
                        self.scene_ids.append(i)
                self.scene_ids = self._generate_unique(self.scene_ids,
                                                       self.hparams.n_objs)
                self.n_matching_scenes = len(self.scene_ids)

        # dataset variables, randomly sample images and labels
        self.img_size = 224
        self.vgg_embedding_dim = 4096

        self.all_features = self._get_noun_features()
        self.all_labels = self._get_noun_labels()
        
        
        available_features = len(self.all_features)
        if self.n_categories > available_features:
            print(f"Warning: n_categories ({self.n_categories}) > available features ({available_features}). Using {available_features} categories.")
            self.n_categories = available_features
            self.hparams.n_categories = available_features

        # Use numpy random generator for sampling
        self.image_ids = self.np_rng.choice(len(self.all_features),
                                          size=min(self.n_categories, len(self.all_features)),
                                          replace=False)
        self.label_ids = self.np_rng.choice(len(self.all_labels),
                                          size=min(self.n_categories, len(self.all_labels)),
                                          replace=False)
        self.features = [self.all_features[id_] for id_ in self.image_ids]
        self.labels = [self.all_labels[id_] for id_ in self.label_ids]

        # generate training set
        if self.hparams.presentation_mode == 'word_object':
            self._generate_word_object_train()
        elif self.hparams.presentation_mode == 'scene_caption':
            self._generate_train()

        # generate evaluation set
        self._generate_eval()

    def save_experiment_data(self, experiment_dir):
        """Save experiment data for this specific run"""
        experiment_name = self.hparams.experiment_name
        
        # Create main experiment directory
        experiment_dir = os.path.join(self.experiment_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save this run's data with run_id in filename
        run_filename = f"run_{self.run_id:02d}.pt"
        run_filepath = os.path.join(experiment_dir, run_filename)
        
        torch.save({
            'train_features': self.train_features,
            'train_caption_ids': self.train_caption_ids,
            'eval_features': self.eval_features, 
            'eval_caption_ids': self.eval_caption_ids,
            'labels': self.labels,
            'original_hparams': self.hparams.__dict__,
            'run_id': self.run_id,
            'image_ids': self.image_ids,  # Save which images were used
            'label_ids': self.label_ids   # Save which labels were used
        }, run_filepath)
        
        print(f"Run {self.run_id:02d} saved to: {run_filepath}")

    def _get_noun_features(self):
        """Get the set of features extracted using VGG16 from the NOUN database"""
        FEATURE_DIR = os.path.join(DATA_DIR, 'features.pickle')
        with open(FEATURE_DIR, 'rb') as f:
            features = pickle.load(f)
        return features

    def _get_noun_labels(self):
        """Retrieve the set of possible nonce labels from the NOUN database"""
        LABEL_DIR = os.path.join(DATA_DIR, 'image_labels.txt')
        with open(LABEL_DIR) as f:
            labels = f.read().splitlines()
        return labels

    def _generate_unique(self, scene_ids, n_objs):
        solved = False
        original_scene_ids = scene_ids.copy()

        while True:
            count = 0
            scene_ids = original_scene_ids.copy()
            new_scene_ids = []

            while not solved:
                scene_ids = np.array(scene_ids)

                # generate decreasing exponential probs
                if self.hparams.dist == 'uniform':
                    p = 1 / (np.ones(len(scene_ids)) * len(scene_ids))
                elif self.hparams.dist == 'zipfian':
                    p = 1 / (np.arange(len(scene_ids)) + 1)
                    p = p / np.sum(p)

                # Use numpy random generator for sampling
                curr_scene_ids = self.np_rng.choice(scene_ids, size=n_objs,
                                                  replace=False, p=p)
                if len(curr_scene_ids) == len(set(curr_scene_ids)):
                    new_scene_ids.append(list(curr_scene_ids))
                    diff = Counter(scene_ids) - Counter(curr_scene_ids)
                    scene_ids = list(diff.elements())

                if len(scene_ids) == 0:
                    solved = True
                    break
                else:
                    count += 1
                    if count % 1000 == 0:
                        print(f"Run {self.run_id}: Attempt {count}")

                if count == 10000:
                    print(f"Run {self.run_id}: Failed to generate unique scenes after 10000 attempts")
                    break

            if solved:
                break

        return new_scene_ids

    def _generate_scene(self, ids):
        """Generate a visual scene from a set of specified ids"""
        features = np.zeros((len(ids), self.vgg_embedding_dim))
        for i, id_ in enumerate(ids):
            features[i] = self.features[id_]

        return None, features

    def _generate_train(self):
        """Generates the set of training trials by converting scene_ids into scenes and captions"""
        train_features = np.zeros((len(self.scene_ids), self.n_objs,
                                self.vgg_embedding_dim))

        # generate scenes and features based on sampled ids
        for i in range(self.n_matching_scenes):
            _, train_features[i] = self._generate_scene(self.scene_ids[i])

        train_caption_ids = self.scene_ids.copy()

        self.train_features = torch.as_tensor(train_features,
                                            dtype=torch.float)
        self.train_caption_ids = torch.as_tensor(train_caption_ids,
                                                dtype=torch.long)

    def _generate_word_object_train(self):
        train_features = np.zeros(
            (len(self.scene_ids) * (self.n_objs ** 2), 1, self.vgg_embedding_dim))
        train_caption_ids = []

        # generate word-object pairs based on scene ids
        idx = 0
        for i in range(self.n_matching_scenes):
            word_object_pairs = list(
                itertools.product(self.scene_ids[i], repeat=2))
            for word_object_pair in word_object_pairs:
                _, train_features[idx] = self._generate_scene([word_object_pair[0]])
                train_caption_ids.append([word_object_pair[1]])
                idx += 1

        self.train_features = torch.as_tensor(train_features,
                                            dtype=torch.float)
        self.train_caption_ids = torch.as_tensor(train_caption_ids,
                                                dtype=torch.long)

    def __getitem__(self, i):
        return self.train_features[i], self.train_caption_ids[i]

    def __len__(self):
        if self.hparams.presentation_mode == 'scene_caption':
            return self.n_matching_scenes
        elif self.hparams.presentation_mode == 'word_object':
            return self.n_matching_scenes * (self.n_objs ** 2)

    def _generate_eval(self):
        """Generate evaluation scenes"""
        assert self.n_foils < self.n_categories

        eval_scene_features = np.zeros((self.n_categories, self.n_foils+1,
                                        self.vgg_embedding_dim))
        eval_caption_ids = []
        eval_captions = []

        for eval_category, i in enumerate(range(self.n_categories)):
            # Use numpy random generator for consistent sampling
            eval_ids = [eval_category] + self.np_rng.choice(np.setdiff1d(np.arange(
                self.n_categories), eval_category), size=self.n_foils, replace=False).tolist()
            
            _, eval_scene_features[i] = self._generate_scene(eval_ids)
            eval_caption_ids.append([eval_category])
            eval_captions.append([self.labels[eval_category]])

        self.eval_features = torch.as_tensor(eval_scene_features, dtype=torch.float)
        self.eval_caption_ids = torch.as_tensor(eval_caption_ids, dtype=torch.long)
        self.eval_captions = eval_captions


def generate_all_runs():
    """Generate all 20 runs and save them in ONE folder"""
    hparams = SimpleNamespace(
        experiment=EXP,
        experiment_name=EXP_NAME,
        n_categories=N_CAT,
        n_objs=N_OBJ,
        n_occurrences=N_OCC,
        n_foils=N_FOILS,
        presentation_mode=PRES,
        dist=DIST,
        seed=SEED
    )
    
    print(f"Generating {N_RUNS} runs for experiment: {EXP_NAME}")
    print(f"All runs will be saved in: experiments/{EXP_NAME}/")
    
    for run_id in range(N_RUNS):
        print(f"\n--- Generating Run {run_id+1}/{N_RUNS} ---")
        
        # Create dataset for this run
        dataset = NounDataset(hparams, BASE_DIR, run_id)
        
        # Save the experiment data for this run
        dataset.save_experiment_data(BASE_DIR)
        
        print(f"Run {run_id+1}/{N_RUNS} completed")
        print(f"  - Training samples: {len(dataset)}")
        print(f"  - Train features shape: {dataset.train_features.shape}")
        print(f"  - Unique images used: {len(dataset.image_ids)}")
    


if __name__ == "__main__":
    generate_all_runs()