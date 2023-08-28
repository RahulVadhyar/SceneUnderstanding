import torch
from tokenizer import Tokenizer
import pandas as pd
from PIL import Image
import torchvision
class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, img_size = (256, 256)):
        self.df = pd.read_csv("/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/archive/flickr30k_images/results.csv", sep = "|")
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
        assert type(idx) == int
        try:
            x = self.df[' comment'][idx][1:]
        except:
            print(f"Issue with {idx}")
            print(self.df[' comment'][idx], idx)
            x = self.df[' comment'][0][1:]

        x = self.tokenizer.encode(x)

        img_path = self.df['image_name'][idx]
        img = Image.open("/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/archive/flickr30k_images/flickr30k_images/" + img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.img_processor(img)
        y = x[1:]
        x = x[:-1]

        x = x + self.sample_output[len(x):]
        y = y + self.sample_output[len(y):]
        if len(x) > 256:
            x = x[:256]
            y = y[:256]

        #convert to tensors and yield
        x = torch.tensor(x).to(torch.int64)
        y = torch.tensor(y).to(torch.int64)
        x = (x, img)
        return x, y