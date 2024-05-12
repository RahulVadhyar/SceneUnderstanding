from model import Foundation
from dataset import CustomDataset
# from flickr import CustomDataset
# from flickr30 import FlickrDataset
import torch
import tqdm
from tokenizer import Tokenizer
import torchvision


BATCH_SIZE = 16
LEARNING_RATE = 5e-4
VOCAB_SIZE = 128
EMBD_SIZE = 768
NUM_HEADS = 64
NUM_BLOCKS = 10
DROPOUT = 0.1
IMG_SIZE = 256
NUM_WORKERS = 2
MODEL_SAVE_DIR = "../models/"
MODEL_SAVE_NAME = "Foundation.pt"

print(f"Using PyTorch version {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.set_float32_matmul_precision('high')



def collate_fn(batch):
    captions = [s[0][0] for s in batch]
    images = [s[0][1] for s in batch]
    images = torch.stack(images)
    captions = torch.stack(captions)
    x = (captions, images)
    y = torch.stack([s[1] for s in batch])
    return x, y

dataset = CustomDataset(img_size = (IMG_SIZE, IMG_SIZE))
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = BATCH_SIZE, 
                                         shuffle = True, 
                                         num_workers = NUM_WORKERS,
                                         collate_fn = collate_fn)

model = Foundation(num_blocks = NUM_BLOCKS,
                   num_heads = NUM_HEADS, 
                   unk_char = dataset.getunk(),
                   vocab_size=VOCAB_SIZE, 
                   embd_size=EMBD_SIZE, 
                   dropout = DROPOUT).to(device)
print(f"The unk char is {dataset.getunk()}")


# model = torch.load(MODEL_SAVE_DIR + MODEL_SAVE_NAME)
optimizer = model.config_optimizer(LEARNING_RATE,)


print(f"The number of parameters is {model.get_num_params()}")
tokenizer = Tokenizer()
tokenizer.load("tokenizer2.pkl")
img_processor = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
])


#training loop
for _ in range(2):
    model.train()
    progress_bar = tqdm.tqdm(dataloader, desc = "COCO dataset")
    i = 0
    for data in progress_bar:
        x, y = data
        x = (x[0].to(device), x[1].to(device))
        y = y.to(device)

        # if i % 1_000== 0:
        #     print(f"Saving model")
        #     torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)
        #     model.eval()
        #     with torch.no_grad():
        #         output = model(x)
        #         output = torch.argmax(output, dim = -1)
        #         print(f"Predicted: {tokenizer.decode(output[0].cpu().numpy())}")
        #         print(f"Actual: {tokenizer.decode(y[0].cpu().numpy())}")
        #     model.train()

        loss, acc = model(x, y, return_loss = True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"
        
        i += 1
    torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)