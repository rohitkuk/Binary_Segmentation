import torch 
import torch.nn as nn 
from tqdm import tqdm

class Train():
    def __init__(self, model, NUM_EPOCHS, loader, criterion, optimizer, Device, grad_scaler=None ):
        self.model = model
        self.epochs = NUM_EPOCHS
        self.loader = loader
        self.criterion= criterion
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.DEVICE = Device

    def __call__(self):
        self.model.train()

        for epoch in range(1,self.epochs):
            tqdm_iter = tqdm(enumerate(self.loader), total = len(self.loader), leave = False)
            epoch_loss = 0
            for batch_idx, (img,mask) in tqdm_iter:
                imgs = img.to(device=self.DEVICE, dtype=torch.float32)
                target = mask.to(device=self.DEVICE, dtype=torch.float)

                if self.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks_pred = self.model(imgs)
                        masks_pred = masks_pred.to(device=self.DEVICE, dtype=torch.float)
                    
                    loss = self.criterion(target, masks_pred) 
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    epoch_loss += loss.item()
                    torch.save(self.model.state_dict(), 'checkpoints/checkpoint_epoch{}.pth'.format(epoch + 1))
                    tqdm_iter.set_description(f"Training Epoch : [{epoch + 1}/{self.epochs}] ")
                    tqdm_iter.set_postfix(
                        batch_loss="%.2f" % loss.item(),
                        epoch_loss= "%.2f" % epoch_loss
                    )
                else:
                    masks_pred = self.model(imgs) 
                    loss = self.criterion(target, masks_pred)   
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    torch.save(self.model.state_dict(), 'checkpoints/checkpoint_epoch{}.pth'.format(epoch + 1))
                    tqdm_iter.set_description(f"Training Epoch : [{epoch + 1}/{self.epochs}] ")
                    tqdm_iter.set_postfix(
                        batch_loss="%.2f" % loss.item(),
                        epoch_loss= "%.2f" % epoch_loss
                    )
                
