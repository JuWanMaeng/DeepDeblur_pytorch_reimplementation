import os
import torch
from tqdm import tqdm
from data import dataset
from model import Model
from loss import Loss
from optim import Optimizer
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import data
import numpy as np

import time





def train():  

    psnr_array=[]
    ssim_array=[]
    save_dir='experiment/weight'
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_state=False
    train_ds=dataset.train_dataset()
    val_ds=dataset.val_dataset()
    

    model=Model()
    model.to(device)

    optimizer=Optimizer(model,scheduler='step')

    criterion=Loss(model=model,optimizer=optimizer)
   

    scaler = amp.GradScaler(
            init_scale=1024,
            enabled=amp_state
        )
    

  
    train_dataloader=DataLoader(train_ds,batch_size=8,num_workers=4,shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,num_workers=4,shuffle=True) 

    

    since = time.time()
    num_epochs=1000
    best_psnr = 0.0

    running_loss=0

    for epoch in range(1,num_epochs):
  
        print('-' * 20)
        print(f'Epoch {epoch}/{num_epochs}    lr{optimizer.get_lr():.2e}')
        

        
    
        model.train()  # Set model to training mode
        criterion.train()
        criterion.epoch=epoch
        

        # Iterate over data.
        
        tq=tqdm(train_dataloader, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
        torch.set_grad_enabled(True)
        for idx,batch in enumerate(tq):
            
    
            inputs, target = data.common.to(
                batch[0], batch[1], device=device, dtype=torch.float32)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            
            with amp.autocast(amp_state):
                output=model(inputs)
               
                loss=criterion(output,target)
                running_loss+=loss

            scaler.scale(loss).backward()
            scaler.step(optimizer.G)
            scaler.update()
            

        criterion.normalize()
        epoch_loss=loss.item()/len(train_dataloader.dataset)
        print(f'loss: {epoch_loss:4f}')

        
            
             
        optimizer.schedule(criterion.get_last_loss())


        if epoch%10 ==0:
            print('@'*20)
            

            model.eval()
            model.to(torch.float32)
            criterion.validate()
            
            tq=tqdm(val_dataloader,ncols=80,smoothing=0,bar_format='val: {desc}|{bar}{r_bar}')

            
            torch.set_grad_enabled(False)
            for idx, batch in enumerate(tq):

                inputs, target = data.common.to(
                                batch[0], batch[1], device=device, dtype=torch.float32)

                with amp.autocast(amp_state):
                    output = model(inputs)


                
                loss=criterion(output, target)
                
                    

                
            
            criterion.normalize()
           
            desc,result=criterion.get_loss_desc()

            epoch_loss=loss.item()/len(train_dataloader.dataset)
            p=result['PSNR']
            s=result['SSIM']
            psnr_array.append(p)
            ssim_array.append(s)
            
            print(f'val_loss: {epoch_loss:4f}, psnr: {p:4f}, SSIM:{s:4f} ')

            if result['PSNR']>best_psnr:
                best_psnr=result['PSNR']
                model.save()



            
            

  



        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best psnr: {:4f}'.format(best_psnr))

    return psnr_array,ssim_array

if __name__ == '__main__':
    psnr_array,ssim_array=train()

    psnr_array=np.array(psnr_array)
    ssim_array=np.array(ssim_array)

    np.save('psnr',psnr_array)
    np.save('ssim',ssim_array)
