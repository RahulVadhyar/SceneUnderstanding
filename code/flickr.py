import torch
from tokenizer import Tokenizer
import json
import pandas as pd
from PIL import Image
import torchvision

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_size = (256, 256)):
        lines = open("/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/flickr8kcaptions.txt", 'r').readlines()
        lines = lines[1:]
        lines = [line.split('.jpg,') for line in lines]
        self.path = "/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/flickr8k/"
        self.df = pd.DataFrame(lines, columns = ['image_id', 'caption'])
        self.tokenizer = Tokenizer()
        self.tokenizer.load("tokenizer2.pkl")
        self.length = len(self.df)
        self.img_size = img_size
        self.sample_output = [self.tokenizer.char_to_idx["[UNK]"] for _ in range(256)]
        self.img_processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length
    
    def getunk(self):
        return self.tokenizer.char_to_idx["[UNK]"]

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        caption = self.df.iloc[idx]['caption']
        x = caption
        x = self.tokenizer.encode(x)

        img_path = self.path + image_id + ".jpg"
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.img_processor(img)
        y = x[1:]
        x = x[:-1]

        x = x + self.sample_output[len(x):]
        y = y + self.sample_output[len(y):]

        #convert to tensors and yield
        x = torch.tensor(x).to(torch.int64)
        y = torch.tensor(y).to(torch.int64)
        x = (x, img)
        return x, y