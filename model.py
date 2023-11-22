import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import datetime
import os

class PoseEstimationNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, cfg.net_emb_dim, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(cfg.net_emb_dim, cfg.net_emb_dim, 5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(cfg.net_emb_dim, cfg.net_emb_dim, 5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(cfg.net_emb_dim, cfg.net_emb_dim, 5)
        self.pool4 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*12*12, cfg.net_emb_dim)
        self.output = nn.Linear(cfg.net_emb_dim, 4*2)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        # print(x.shape)
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.tanh(self.output(x))
        # print(x.shape)
        return x
    
    
### Training code
def retrain(model, cfg):
    # load model
    
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # load data
    im_path = cfg.data_path+'images/'
    import csv
    annotations = {}
    with open(cfg.data_path+'annotations.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['file_name'] not in annotations:
                annotations[row['file_name']] = []
            annotations[row['file_name']] += [[row['x'],row['y']]]
    # print(annotations)
    
    for key_fn, value_xy in annotations.items():
        # print(im_path+key_fn)
        w,h = image.size
        
        points = np.array(value_xy, dtype=np.float32)
        points[:,0] *=  cfg.net_inputs[0]/w
        points[:,1] *=  cfg.net_inputs[1]/h
        points = points.astype(np.int32).astype(np.float32)
        points[:,0] /= cfg.net_inputs[0]
        points[:,1] /= cfg.net_inputs[1]
        target = torch.tensor(points).float()
        
        image = Image.open(im_path+key_fn)
        image = image.resize(cfg.net_inputs)
        image = torch.tensor(np.array(image)).float()
        image = image.permute(2,0,1)
        
        opt.zero_grad()
        output = model(image)
        output = output.reshape((4,2))
        
        loss = criterion(target, output)
        loss.backward()
        opt.step()
        
        print('Loss: {:.6f}'.format(loss.item()))

    # save model
    postfix = datetime.date.today().isoformat()
    os.makedirs(cfg.model_weight_path, exist_ok=True)
    torch.save(model.state_dict(), cfg.model_weight_path+f'weights_{postfix}.tensor')
