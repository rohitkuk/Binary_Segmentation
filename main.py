
from src.dataset import HumanSegmentation
from src.model import UNet
from src.customTransforms import IMG_Trasforms, MASK_Trasforms
from src.train import Train



import torch 
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import DataLoader



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = 10
LEARNING_RATE= 2e-4
BATCH_SIZE   = 1


dataset_ = HumanSegmentation(IMG_Trasforms, MASK_Trasforms)
training_loader = DataLoader(dataset_, batch_size = BATCH_SIZE)
model = UNet(3,1).to(DEVICE)

optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

criterion = nn.BCEWithLogitsLoss()

grad_scaler = None

if DEVICE == "cuda":
    grad_scaler = torch.cuda.amp.GradScaler()

Train_ = Train(model, NUM_EPOCHS, training_loader, criterion, optimizer, DEVICE, grad_scaler)
Train_()

