import torch
import torchvision
from tokenizer import Tokenizer
import json
import pandas as pd
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_size = (256, 256)):
        metadata = json.load(open("../dataset/annotations/captions_train2014.json"))
        self.annotations = metadata['annotations']
        self.img_data = metadata['images']
        self.img_data = pd.DataFrame(self.img_data)
        self.tokenizer = Tokenizer()
        self.tokenizer.load("tokenizer2.pkl")
        self.length = len(self.annotations)
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
        annotation = self.annotations[idx]
        img_id = annotation['image_id']
        x = annotation['caption']
        x = self.tokenizer.encode(x)

        img_path = self.img_data.loc[self.img_data['id'] == img_id]['file_name'].values[0]
        img = Image.open("../dataset/train2014/" + img_path)
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