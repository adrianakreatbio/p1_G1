# training datasets

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import imageio.v3 as iio
import numpy as np
import os
import torch.nn.functional as F
from u_net import GelUNet, preprocess_gel

# 1434 begins 1
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
# 1434 ends 1



"""
    def __getitem__(self, idx):
        fname = self.files[idx]

        # load images
        img = iio.imread(os.path.join(self.img_dir, fname))         # read .tif images as 2D numpy array w shape >
        img = np.array(img, copy=True)                             # force a real copy so its storage is resizable
        img_t = preprocess_gel(img)                                # change from [[0-255,0-255],[0-255,0-255]] (r>

        # load masks
        mask = iio.imread(os.path.join(self.mask_dir, fname))       # (H, W)
        mask = (mask > 0).astype(np.float32)                        # turns all numbers from 0-255 in mask into b>
        mask_t = torch.from_numpy(mask)[None, ...]                  # change from numpy (H,W) into (1, H, W), 1 i>

        # remove extra batch dimension (DataLoader will add it)
        return img_t.squeeze(0), mask_t                             # removes the batchif its size = 1 = (1,H,W)
"""



### Get Dataset
class GelSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # read as grayscale, no EXIF
        img_path  = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)  # change ext if needed

        img  = _read_gray(img_path)                 # uint8 (H,W)
        mask = _read_gray(mask_path)                # uint8 (H,W)
        mask = (mask > 127).astype(np.float32)      # binary 0/1 float32

        # to tensors
        img_t  = preprocess_gel(img).squeeze(0).clone()      # (1,H,W) float
        mask_t = torch.from_numpy(mask).unsqueeze(0).clone() # (1,H,W) float

        # resize
        target_size = (256, 256)
        img_t  = F.interpolate(img_t.unsqueeze(0),  size=target_size, mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=target_size, mode="nearest").squeeze(0)

        return img_t, mask_t



def train_model(img_dir, mask_dir, file_list, num_epochs=20, lr=1e-3, batch_size=4, out_file="trained_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")               # decide whether to run on cuda (installed)=NVIDIA GPU (faster deep learning) or CPU (small dataset)
    dataset = GelSegDataset(img_dir, mask_dir, file_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)    # 2 num_workers = 2 CPU work in parallel
    model = GelUNet(in_channels=1, out_channels=1).to(device)       # run model on GPU/CPU
    criterion = nn.BCEWithLogitsLoss()                              # loss fxn: combines sigmoid activation & binary cross-entropy loss to measure how close the output is to the true mask (0/1). Lower loss = better match.                             
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # Adam algorithm: reduce loss so logits for band is high +ve, background is low -ve. Update weights based on loss gradient, w learning rate/lr controlling how big each update step is.
                                                                    # num_epochs=20 means the model trains through the entire dataset 20 times (more epochs = more learning, but too many can cause overfitting - memorizing but not generalizing)
    for epoch in range(num_epochs):                                 # repeat training 20x. epoch = model sees every training image once. if 500 images, batch size=10, so 1 epoch = 50 batches.
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)                                    # insert images into model & train
            loss = criterion(logits, masks)                         # calculate the loss between logits and true masks
            loss.backward()                                         # calculate gradients of the loss w respect to every weight using backpropagation
            optimizer.step()                                        # update all weights based on those gradients - learning
            total_loss += loss.item() * imgs.size(0)                # accumulate total loss for this batch * #images

        avg_loss = total_loss / len(loader.dataset)                 # mean loss per image. So all these is to calculate how confident the model's yes/no decision is.
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")   # So yes — high variation in gel intensity leads to higher loss early, but the model will normalize and learn robust band boundaries as training progresses.

    torch.save(model.state_dict(), out_file)                        # save trained dataset in state_dict()
    print(f"Training done. Weights saved to {out_file}")


# 1434 starts 1
def _read_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")          # bypass EXIF autorotate path
        return np.array(im, dtype=np.uint8)
# 1434 ends 2


if __name__ == "__main__":
    img_dir = "train/images"
    mask_dir = "train/masks"
    file_list = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    train_model(img_dir, mask_dir, file_list, num_epochs=20, lr=1e-3, batch_size=4)


""" IMAGE FORMAT
e.g.
raw image format for 4x4 pixel image, array's shape = (4,4), showing intensity values (if 8-bit = 0-255)
[[  0, 120, 200, 255],
 [ 10, 150, 210, 240],
 [  5, 140, 180, 220],
 [  0, 100, 160, 200]]

tensor format: converts to float32 > removes background > divides by max val to scale to [0,1] > adds 2 extra dimensions = (1,1,4,4) MEANS (1 image, 1 channel, 4 pi height, 4 pi width), showing intensity vals in float
tensor([[[[0.00, 0.47, 0.78, 1.00],
          [0.04, 0.59, 0.82, 0.94],
          [0.02, 0.55, 0.70, 0.86],
          [0.00, 0.39, 0.63, 0.78]]]])
"""

"""MASK FORMAT
 from pixel intensities,
[[  0, 120, 200, 255],
 [ 10, 150, 210, 240],
 [  5, 140, 180, 220],
 [  0, 100, 160, 200]]

 to binary,
 [[0, 0, 1, 1],
 [0, 1, 1, 0],
 [0, 0, 0, 0],
 [1, 0, 1, 0]]
"""

# Sigmoid activation fxn: converts raw logits to y(0,1), 0 = strong no, 1 = strong yes. In the curve, x=logits [2.0, -1.0, 0.0], y=sigmoid val/probability 0 to 1 (closer to 0 = no, closer to 1 = yes)
# Binary Cross-Entropy, BCE (measures how far those prob are from true labels 0/1). formula for average loss: (sum of -log(sigmoid(x)) if y=1, -log(sigmoid(x)) if y=0)/3    


### TO-DO
# 1 Prepare gel images & masks on ImageJ or Fiji or CVAT or Photoshop:  real photo & manually drawn white bands (0) w black background (0)
# 2 Check tif format

