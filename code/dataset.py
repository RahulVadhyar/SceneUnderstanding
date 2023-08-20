import torch
import torchvision
from tokenizer import Tokenizer
import json
import pandas as pd
from PIL import Image
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_size = (256, 256)):
        metadata = json.load(open("/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/annotations_trainval2014/annotations/captions_train2014.json"))
        self.annotations = metadata['annotations']
        self.img_data = metadata['images']
        self.img_data = pd.DataFrame(self.img_data)
        self.tokenizer = Tokenizer()
        self.tokenizer.load("tokenizer2.pkl")
        self.length = len(self.annotations)
        self.img_size = img_size
        self.totensor = torchvision.transforms.ToTensor()
        self.sample_output = [self.tokenizer.char_to_idx["[UNK]"] for _ in range(256)]

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
        img = Image.open("/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/train2014/" + img_path)
        img = img.resize((256, 256))
        img = np.array(img)
        img = self.totensor(img)
        img = img/255.
        if img.shape[0] == 1:
            img = torch.stack([img[0], img[0], img[0]])
        

        y = x[1:]
        x = x[:-1]

        x = x + self.sample_output[len(x):]
        y = y + self.sample_output[len(y):]

        #convert to tensors and yield
        x = torch.tensor(x).to(torch.int64)
        y = torch.tensor(y).to(torch.int64)
        x = (x, img)
        return x, y