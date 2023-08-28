from model import Foundation
from dataset import CustomDataset
# from flickr import CustomDataset
from flickr30 import FlickrDataset
import torch
import tqdm
from PIL import Image
from tokenizer import Tokenizer
import torchvision

BATCH_SIZE = 16
LEARNING_RATE = 1e-6
VOCAB_SIZE = 128
EMBD_SIZE = 256
NUM_HEADS = 16
NUM_BLOCKS = 8
DROPOUT = 0.1
IMG_SIZE = 384
NUM_WORKERS = 2

CSV_DIR = "/home/starsystem/Documents/SceneUnderstanding/models/"
MODEL_SAVE_DIR = "/home/starsystem/Documents/SceneUnderstanding/models/"
MODEL_SAVE_NAME = "Foundation.pt"
CSV_NAME = "Foundation.csv"

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
flickr_dataset = FlickrDataset(img_size = (IMG_SIZE, IMG_SIZE))
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = BATCH_SIZE, 
                                         shuffle = True, 
                                         num_workers = NUM_WORKERS,
                                         collate_fn = collate_fn)
flickr_dataloader = torch.utils.data.DataLoader(flickr_dataset,
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

# model = torch.load(MODEL_SAVE_DIR + MODEL_SAVE_NAME)
optimizer = model.config_optimizer(1e-3, 1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


print(f"The number of parameters is {model.get_num_params()}")
tokenizer = Tokenizer()
tokenizer.load("tokenizer2.pkl")
img_processor = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
])

def evaluate():
    #griraffee bending and eating grass and someting
    img_path = "/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/test.jpg"
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img_processor(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    text = [tokenizer.char_to_idx["[START]"]]
    text = torch.tensor(text).to(torch.int64).to(device)
    # print(text.shape)

    model.eval()
    with torch.no_grad():
        i = 0
        while True:
            output = model((text.unsqueeze(0), img))
            output = output.argmax(dim = -1)
            text = torch.concat([text, output[-1][-1].unsqueeze(0)], dim = -1)
            if output[0][-1] == tokenizer.char_to_idx["[END]"]:
                break
            i += 1
            if i > 1000:
                break
    model.train()
    print("The validation string is:", end = " ")
    print(tokenizer.decode(text[1:-1].tolist()))


#training loop

model.train()
progress_bar = tqdm.tqdm(dataloader, desc = "COCO dataset")
i = 0
for data in progress_bar:
    x, y = data
    x = (x[0].to(device), x[1].to(device))
    y = y.to(device)
    loss, acc = model(x, y, return_loss = True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"
    if i % 1_000== 0:
        print(f"Saving model")
        torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)
    i += 1
evaluate()
torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)

for _ in range(5):
    model.train()
    progress_bar = tqdm.tqdm(flickr_dataloader, desc = "Flickr dataset")
    i = 0
    for data in progress_bar:
        x, y = data
        x = (x[0].to(device), x[1].to(device))
        y = y.to(device)
        loss, acc = model(x, y, return_loss = True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"
        if i % 1_000== 0:
            print(f"Saving model")
            torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)
        i += 1
    evaluate()
    torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)