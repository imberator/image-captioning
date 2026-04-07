import os
import torch
import pandas as pd
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from collections import Counter
from torch.utils.data import Dataset

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import kagglehub


def download_and_parse_captions():
    """Download the Flickr8k dataset (or resolve from cache) and parse captions.txt.
    
    Returns:
        data_dir: Absolute path to the dataset root directory.
        get_captions: Dict mapping image filename -> list of caption strings.
        all_captions: Flat list of every caption string.
    """
    print("Resolving dataset path from Kaggle cache...")
    data_dir = kagglehub.dataset_download("adityajn105/flickr8k")

    caption_filename = os.path.join(data_dir, "captions.txt")

    get_captions = {}
    all_captions = []

    if os.path.exists(caption_filename):
        with open(caption_filename, 'r') as f:
            lines = f.readlines()

        # Kaggle's Flickr8k 'captions.txt' has a header row
        start_idx = 1 if "image,caption" in lines[0] else 0

        for caption in lines[start_idx:]:
            caption = caption.rstrip("\n")
            
            if not caption:
                continue
                
            data = caption.split(".jpg,")
            if len(data) != 2:
                continue
                
            img_name = data[0] + ".jpg"
            
            caption_list = get_captions.get(img_name, [])
            caption_list.append(data[1])
            get_captions[img_name] = caption_list

            all_captions.append(data[1])
            
        print(f"Successfully loaded {len(get_captions)} unique images and {len(all_captions)} captions from {data_dir}/.")
    else:
        print(f"[!] Dataset not found at '{caption_filename}'.")
        print(f"    Run the script once with internet access so kagglehub can download it.")

    return data_dir, get_captions, all_captions


def build_vocabulary(all_captions):
    """Build a torchtext vocabulary from a list of caption strings.
    
    Returns:
        vocabulary: torchtext Vocab object with special tokens inserted.
        word_tokenizer: The tokenizer function used to split captions.
    """
    word_tokenizer = get_tokenizer("basic_english")
    vocab_frequency = Counter()

    for caption in all_captions:
        vocab_frequency.update(word_tokenizer(caption))

    vocabulary = vocab(vocab_frequency)
    vocabulary.insert_token("<UNKNOWN>", 0)
    vocabulary.insert_token("<PAD>", 1)
    vocabulary.insert_token("<START>", 2)
    vocabulary.insert_token("<END>", 3)
    vocabulary.set_default_index(0)

    return vocabulary, word_tokenizer


class ImageCaptioningDataset(Dataset):
    """PyTorch Dataset for Flickr8k image-caption pairs."""

    def __init__(self, df_split, split, data_dir, vocabulary, word_tokenizer, context_length):
        self.df = df_split
        self.data_dir = data_dir
        self.vocabulary = vocabulary
        self.word_tokenizer = word_tokenizer
        self.context_length = context_length
        self.img_size = 224

        transformation_list = [alb.Resize(self.img_size, self.img_size)]
        if split == "training":
            transformation_list.append(alb.HorizontalFlip())
            transformation_list.append(alb.ColorJitter())
        transformation_list.append(alb.Normalize())
        transformation_list.append(ToTensorV2())

        self.transformations = alb.Compose(transformation_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filename, captions = self.df.iloc[idx]
        
        img_path = os.path.join(self.data_dir, "Images", image_filename)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Could not find image at {img_path}. Make sure it is extracted.")
            
        actual_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        transformed_img = self.transformations(image=actual_image)["image"]

        encoded_captions = []
        for caption in captions:
            tokens = self.word_tokenizer(caption)
            integers = [self.vocabulary[token] for token in tokens]

            # Adding start and end tokens
            integers = [2] + integers + [3]

            if len(integers) <= self.context_length:
                pads_to_add = self.context_length - len(integers)
                integers += [1] * pads_to_add
            else:
                integers = integers[:self.context_length - 1] + [3]

            encoded_captions.append(torch.tensor(integers, dtype=torch.long))
        
        random_idx = torch.randint(len(encoded_captions), (1,)).item()
        return transformed_img, encoded_captions[random_idx]
