import gym
import numpy as np
from gym import spaces
import utils
import os
import cv2
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    RandomRotation,
                                    ColorJitter,
                                    Resize, 
                                    ToTensor)
import torch
import torch.nn as nn
import random
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import argparse
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def get_attention_map(img, size, model):
    model = model.cuda()
    x = img.cuda()
    # x.size()

    results = model(x, output_attentions=True)
    logits, att_mat = results.logits, results.attentions
    # prediction = torch.argmax(logits, 1)

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).cuda()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
     
    mask = torch.tensor(cv2.resize(mask / mask.max(), size)[..., np.newaxis]).permute(2, 0, 1)
    # result = (mask * img[0]).type(torch.uint8)
    return mask

class SelectEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, N_IMAGE, N_SELECTION, random_seed=42):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.N_IMAGE = N_IMAGE
        self.N_SELECTION = N_SELECTION
        self.action_space = spaces.Discrete(self.N_IMAGE)
        # Example for using image as input (channel-first; channel-last also works):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch32-384')
        self.HEIGHT, self.WIDTH = self.processor.size['height'], self.processor.size['width']
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.N_IMAGE, self.HEIGHT, self.WIDTH), dtype=np.uint8)

        root_dir = 'data/POAAGG'
        patient_cases = []
        folders = {}
        for item in os.listdir(os.path.join(root_dir, 'cases')):
            patient = item.split('_')[0]
            d = os.path.join(root_dir, 'cases', item)
            if os.path.isdir(d):
                patient_cases.append(patient)
                if patient in folders:
                    folders[patient].append(d)
                else:
                    folders[patient] = [d]

        patient_controls = []
        for item in os.listdir(os.path.join(root_dir, 'controls')):
            patient = item.split('_')[0]
            d = os.path.join(root_dir, 'controls', item)
            if os.path.isdir(d):
                patient_controls.append(patient)
                if patient in folders:
                    folders[patient].append(d)
                else:
                    folders[patient] = [d]

        patients = list(set(patient_cases + patient_controls))
        patients.sort()
        train_idx,test_idx,val_idx = torch.utils.data.random_split(patients, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
        train_patient = [patients[i] for i in train_idx.indices]
        val_patient = [patients[i] for i in val_idx.indices]
        test_patient = [patients[i] for i in test_idx.indices]
        
        self.train_folders = []
        for patient in train_patient:
            train_folder = folders[patient]
            self.train_folders.extend(train_folder)

        self.val_folders = []
        for patient in val_patient:
            val_folder = folders[patient]
            self.val_folders.extend(val_folder)
            
        self.test_folders = []
        for patient in test_patient:
            test_folder = folders[patient]
            self.test_folders.extend(test_folder)
                
        self.indices = []
        self.selected = []
        # self.feature_extractor = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')
        self.encoder = ViTForImageClassification.from_pretrained('./models/test-POAAGG-filter')

    def step(self, action):
        done = False
        self.select += 1
        if self.select > self.N_SELECTION:
            done = True
        if action in self.dummies:
            reward = -10
        elif action in self.selected:
            reward = -1
        elif action in self.indices:
            reward = 10
            self.selected.append(action)
        else:
            reward = 0

        return self.observation, reward, done, {}

    def reset(self, train=True):
        if train:
            folder = random.choice(self.train_folders)
        else:
            folder = random.choice(self.val_folders)
        paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
        images = [os.path.join(folder, path) for path in paths if path[-3:].lower() in ['jpg', 'png', 'tif']]
        phoneNumRegex = re.compile(r'P(\d+)_P(\d+)')
        indices = []
        for path in paths:
            try:
                groups = phoneNumRegex.search(path).groups()
                indices.extend(groups)
            except:
                pass
        indices = list(set(indices))
        self.indices = [int(i) for i in indices]
        observation = []
        real_count = 0
        for count, image in enumerate(images):
            if count == self.N_IMAGE:
                break
            # try: 
            img = Image.open(image)
            img = self.processor(img, return_tensors="pt")['pixel_values']
            size = (self.HEIGHT, self.WIDTH)
            attention = get_attention_map(img=img, model=self.encoder, size=size)
            observation.append(attention)
            # observation.append(img[0])
            real_count += 1
            # except:
            #     pass
        dummies = []
        if real_count < self.N_IMAGE:
            for i in range(self.N_IMAGE - real_count):
                dummies.append(count+1+i)
                observation.append(torch.zeros(1, 384, 384))
        observation = torch.vstack(observation)
        self.observation = observation
        self.select = 0
        self.selected = []
        self.dummies = dummies

        return observation  # reward, done, info can't be included


    def render(self, mode="human"):
        pass

    def close(self):
        pass

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    env = SelectEnv(N_IMAGE=10, N_SELECTION=6)
    observations = env.reset()
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    if args.train:
        model.learn(total_timesteps=25000)
        model.save("sample_selection")

    model = PPO.load("sample_selection")
    obs = env.reset(train=False)
    rewards_list = []
    reward = 0
    total_count = 100
    count = 0
    while count < total_count:
        # print(count)
        action, _states = model.predict(obs.numpy())
        obs, rewards, dones, info = env.step(action)
        reward += rewards
        if dones:
            obs = env.reset(train=False)
            rewards_list.append(reward)
            reward = 0
            count += 1
    print(rewards_list)
    print('Average', np.mean(rewards_list))