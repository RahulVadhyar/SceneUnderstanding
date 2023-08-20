from model import Foundation
from dataset import CustomDataset
import torch
import tqdm
from PIL import Image
import torchvision
from tokenizer import Tokenizer
import math

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
VOCAB_SIZE = 128
EMBD_SIZE = 512
NUM_HEADS = 32
NUM_BLOCKS = 16
DROPOUT = 0.2
IMG_SIZE = 256
NUM_WORKERS = 4
WARMUP_STEPS = 400

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

# model = torch.load(MODEL_SAVE_DIR + MODEL_SAVE_NAME)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

print(f"The number of parameters is {model.get_num_params()}")
# model = torch.compile(model)
totensor = torchvision.transforms.ToTensor()
tokenizer = Tokenizer()
tokenizer.load("tokenizer2.pkl")

def evaluate():
    #griraffee bending and eating grass and someting
    img_path = "/home/starsystem/Documents/SceneUnderstanding/dataset(coco)/val2014/COCO_val2014_000000001448.jpg"
    raw_img = Image.open(img_path).resize((256, 256))
    img = totensor(raw_img)
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
            # print(text)
            if output[0][-1] == tokenizer.char_to_idx["[END]"]:
                break
            i += 1
            if i > 1000:
                break
    model.train()
    print("The validation string is:", end = " ")
    print(tokenizer.decode(text[1:-1].tolist()))


#training loop

for epoch in range(20):
    model.train()
    progress_bar = tqdm.tqdm(dataloader, desc = f"Epoch {epoch + 1}/20")
    i = 0
    evaluate()
    for data in progress_bar:
        x, y = data
        x = (x[0].to(device), x[1].to(device))
        y = y.to(device)
        loss = model(x, y, return_loss = True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress_bar.postfix = f"Loss: {loss.item()}"
        if i % 1_000== 0:
            print(f"Saving model")
            torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)
        i += 1

    torch.save(model, MODEL_SAVE_DIR + MODEL_SAVE_NAME)

