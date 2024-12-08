from torchvision import transforms
import cv2
import torch
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Data science tools
import numpy as np
import pandas as pd
import os
from skimage import io
from PIL import Image

# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Other imports
import segmentation_models_pytorch as smp
from os import path
from importlib import import_module
from tqdm import tqdm



class SegData(Dataset):
    
    def __init__(self, root_dir, images_path , masks_path, img_names, h, w, mean, std, in_channels, out_channels, return_path=False, ONN=True):
        self.root_dir = root_dir
        self.images_path = images_path
        self.masks_path = masks_path 
        self.img_names = img_names
        self.h = h # image height 
        self.w = w # image width
        self.mean = mean
        self.std = std 
        self.ONN = ONN
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_path = return_path
        self.conv_fact = 1 
        # conversion factor to convert class values from np.linspace(0,255,out_channels) to np.linspace(0,out_channels-1,out_channels)
        if self.out_channels > 1:
            self.conv_fact = 255

        if self.ONN:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
        else:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std) 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std)  
                    ]) 
        if self.out_channels==1:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((self.h,self.w)),
                transforms.ToTensor(),  
                ]) 
        else:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.h,self.w), interpolation = transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),  
                ]) 
    
    def __len__(self): 
        return len(self.img_names)
    
    def __getitem__(self, index): 
        # read 
        image = Image.open(os.path.join(self.root_dir, self.images_path+'/'+self.img_names[index]))
        image = np.array(image)
        mask  = Image.open(os.path.join(self.root_dir, self.masks_path+ '/'+self.img_names[index]))
        mask = np.array(mask)
        
        # apply transformation  
        image = self.img_transforms(image)
        if self.ONN:
            image = 2.0*image -1  
        # image = 2.0*image -1 
        # image = image-self.mean / self.std
        mask = self.conv_fact*self.mask_transforms(mask)
        mask = mask.long() 
        # mask= 1*(mask>=0.5)  
        if self.return_path: 
            return image, mask, self.img_names[index]
        else:
            return image, mask


class TestData(Dataset): 
    
    def __init__(self, root_dir, images_path, img_names, h, w, mean, std, in_channels, return_path=False, ONN=True):
        self.root_dir = root_dir
        self.images_path = images_path
        self.img_names = img_names
        self.h = h # image height 
        self.w = w # image width
        self.mean = mean
        self.std = std 
        self.ONN = ONN
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
        self.in_channels = in_channels
        self.return_path = return_path 
        if self.ONN:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
        else:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std) 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std)  
                    ]) 

    def __len__(self):
        return len(self.img_names)
      
    def __getitem__(self, index):
        # read image
        #image = io.imread(os.path.join(self.root_dir, self.images_path+'/'+self.img_names[index]))
        image = Image.open(os.path.join(self.root_dir, self.images_path+'/'+self.img_names[index]))
        image = np.array(image)
        # apply transformation
        image = self.img_transforms(image)
        if self.ONN:
            image = 2.0*image -1  
        if self.return_path:
            return image, self.img_names[index] 
        else: 
            return image


# import torch.nn.functional as F
# ohe = F.one_hot(labels, num_classes)

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    if labels.is_cuda:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    else:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    assert labels.is_cuda == target.is_cuda, 'target & labels disagree on CUDA status'
    return target

def IoU_training(target,output,threshold):
    smooth = 1.0
    output = 1*(output>=threshold)  # if output[i] >=0.5 then set to 1 else 0
    target = 1*(target>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
    TP = torch.sum((output==1) & (target==1)).double()
    FP = torch.sum((output==1) & (target==0)).double() 
    FN = torch.sum((output==0) & (target==1)).double() 
    IoU = (TP+smooth)/(TP+FP+FN+smooth) 
    return IoU.item()

def DSC_training(target,output,threshold):
    smooth = 1.0
    output = 1*(output>=threshold)  # if output[i] >=0.5 then set to 1 else 0
    target = 1*(target>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
    output = output.reshape((output.size(0),-1))
    target = target.reshape((target.size(0),-1))
    intersection = (output * target).sum(1)
    DSC = (2. * intersection + smooth) / (output.sum(1) + target.sum(1) + smooth)
    return DSC.mean().item()


def compute_IoU(target,output,threshold):
    IoU = 0.0
    smooth = 1
    for i in range(target.shape[0]):
        temp_o = 1*(output[i]>=threshold)  # if output[i] >=0.5 then set to 1 else 0
        temp_g = 1*(target[i]>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
        TP = torch.sum((temp_o==1) & (temp_g==1)).double()
        FP = torch.sum((temp_o==1) & (temp_g==0)).double() 
        FN = torch.sum((temp_o==0) & (temp_g==1)).double() 
        temp_IoU = (TP+smooth)/(TP+FP+FN+smooth) 
        # temp_IoU = (TP)/(TP+FP+FN+1e-9)
        IoU += temp_IoU 
    return IoU.item()/target.shape[0] 

def compute_DSC(target,output,threshold):
    DSC = 0.0   
    smooth = 1
    for i in range(target.shape[0]):
        temp_o = 1*(output[i]>=threshold)  # if output[i] >=0.5 then set to 1 else 0
        temp_g = 1*(target[i]>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
        TP = torch.sum((temp_o==1) & (temp_g==1)).double()
        FP = torch.sum((temp_o==1) & (temp_g==0)).double() 
        FN = torch.sum((temp_o==0) & (temp_g==1)).double() 
        temp_DSC = (2*TP+smooth)/(2*TP+FP+FN+smooth)  
        # temp_DSC = (2*TP)/(2*TP+FP+FN+1e-9) 
        DSC += temp_DSC 
    return DSC.item()/target.shape[0]


def Multi_Class_DSC(target,output,C,threshold, disregard_background = True):
    """
    Computes a Dice  from 2D input of class scores and a target of integer labels.

    Parameters
    ----------
    input : 
        size B x C x H x W representing class scores.
    target : 
        integer label representation of the ground truth, same size as the input.
        size B x 1 x H x W representing class scores, where each value, 0 < label_i < C-1

    Returns
    -------
    dice_total : float.
    total dice 
    """
    # output = F.softmax(output, dim=1)
    # target = F.one_hot(target, C)
    # target = target.permute(0,1,3,2).permute(0,2,1,3)

    # DSC = torch.zeros((C-1,1))
    # for clas_i in range(1,C):
    #     DSC[clas_i-1,0]  = DSC_training((target[:,clas_i,:,:]),(output[:,clas_i,:,:]),threshold)

    output = F.softmax(output, dim=1)
    _, output = torch.max(output, dim=1)
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = C)
    DSC = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        DSC = DSC[:, 1:]
    #return DSC.mean().item()
    return DSC.mean().item()


def calc_acc(target,output,threshold):
    target = 1*(target>=threshold)  # if target >=0.5 then set to 1 else 0
    output = 1*(output>=threshold)  # if output >=0.5 then set to 1 else 0
    return torch.mean((target==output).float()) 


def Createlabels(datadir, Seg_state=False):
    categories = []
    n_Class = [] 
    img_names = []
    labels = []
    i = 0
    for d in os.listdir(datadir):
        categories.append(d)
        if Seg_state:
            if i==1:
                break
        temp = os.listdir(datadir + d)
        img_names.extend(temp) 
        n_temp = len(temp)
        if i==0:
            labels = np.zeros((n_temp,1)) 
        else:
            labels = np.concatenate( (labels, i*np.ones((n_temp,1))) )
        i = i+1
        n_Class.append(n_temp)
    if Seg_state:
        return categories, n_Class, img_names 
    else:
        return categories, n_Class, img_names, labels 
    


def train(
    out_channels,
    model,
    criterion,
    optimizer,
    lossType,
    scheduler,
    train_loader,
    valid_loader,
    test_loader,
    Test_Mask,
    threshold,
    save_file_name,
    train_on_gpu,
    history=[],
    max_epochs_stop=5,
    start_epoch=0,
    n_epochs=30,
    print_every=2,
    disregard_background=True,
    use_custom_dataloader=False  # Added parameter
):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): CNN to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizer): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        test_loader (PyTorch dataloader): testing dataloader
        Test_Mask (bool): whether test masks are available
        threshold (float): threshold for segmentation output
        save_file_name (str): file path to save the model state dict
        train_on_gpu (bool): whether training is on GPU
        history (list): history of losses and metrics
        max_epochs_stop (int): maximum number of epochs with no improvement for early stopping
        start_epoch (int): starting epoch number
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
        disregard_background (bool): whether to disregard background class in metrics
        use_custom_dataloader (bool): whether to use custom dataloader for dual modality inputs

    Returns
    --------
        model (PyTorch model): trained CNN with best weights
        history (DataFrame): history of train and validation loss and metrics
    """

    # Early stopping initialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_best_DSC = 0
    best_epoch = 0

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(start_epoch, n_epochs):

        # Keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        train_DSC = 0
        valid_DSC = 0
        test_DSC = 0

        # Set to training
        model.train()
        start = timer()
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        # Training loop
        for ii, data in enumerate(pbar):
            # Handle data unpacking based on custom dataloader
            if use_custom_dataloader:
                (inputs_t1c, inputs_t2f), target = data
                if train_on_gpu:
                    inputs_t1c = inputs_t1c.to('cuda', non_blocking=True)
                    inputs_t2f = inputs_t2f.to('cuda', non_blocking=True)
                    target = target.to('cuda', non_blocking=True)
                inputs = (inputs_t1c, inputs_t2f)
            else:
                inputs, target = data
                if train_on_gpu:
                    inputs = inputs.to('cuda', non_blocking=True)
                    target = target.to('cuda', non_blocking=True)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs
            if use_custom_dataloader:
                output = model(*inputs)  # Unpack inputs
            else:
                output = model(inputs)

            # Loss and backpropagation of gradients
            if lossType == 'CrossEntropy':
                loss = criterion(output, target.squeeze(1))
            elif lossType == 'BCELoss':
                output = torch.sigmoid(output)
                loss = criterion(output.squeeze(1), target.float().squeeze(1))
            elif lossType == 'DiceLoss':
                output = torch.sigmoid(output)
                loss = criterion(output, target)
            elif lossType == 'CompoundLoss':
                output = torch.sigmoid(output)
                loss = criterion(output, target)
            elif 'SMP' in lossType:
                loss = criterion(output, target.squeeze(1))

            loss.backward()
            # Update the parameters
            optimizer.step()
            # Accumulate loss
            train_loss += loss.item() * target.size(0)
            # Calculate DSC
            if out_channels == 1:
                DSC = DSC_training(target, output, threshold)
            else:
                DSC = Multi_Class_DSC(target.squeeze(1), output, out_channels, threshold, disregard_background)
            train_DSC += DSC * target.size(0)
            pbar.set_postfix(loss=loss.item(), DSC=DSC)
            # Release memory
            del output, target, data, inputs, loss, DSC

        # After training loop ends, start validation
        else:
            model.epochs += 1
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                pbar_val = tqdm(valid_loader, ncols=100, desc='Validating', leave=False)
                for _, data in enumerate(pbar_val):
                    if use_custom_dataloader:
                        (inputs_t1c, inputs_t2f), target = data
                        if train_on_gpu:
                            inputs_t1c = inputs_t1c.to('cuda', non_blocking=True)
                            inputs_t2f = inputs_t2f.to('cuda', non_blocking=True)
                            target = target.to('cuda', non_blocking=True)
                        inputs = (inputs_t1c, inputs_t2f)
                    else:
                        inputs, target = data
                        if train_on_gpu:
                            inputs = inputs.to('cuda', non_blocking=True)
                            target = target.to('cuda', non_blocking=True)

                    # Forward pass
                    if use_custom_dataloader:
                        output = model(*inputs)
                    else:
                        output = model(inputs)

                    # Loss
                    if lossType == 'CrossEntropy':
                        loss = criterion(output, target.squeeze(1))
                    elif lossType == 'BCELoss':
                        output = torch.sigmoid(output)
                        loss = criterion(output.squeeze(1), target.float().squeeze(1))
                    elif lossType == 'DiceLoss':
                        output = torch.sigmoid(output)
                        loss = criterion(output, target)
                    elif lossType == 'CompoundLoss':
                        output = torch.sigmoid(output)
                        loss = criterion(output, target)
                    elif 'SMP' in lossType:
                        loss = criterion(output, target.squeeze(1))

                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * target.size(0)
                    # Calculate validation DSC
                    if out_channels == 1:
                        DSC = DSC_training(target, output, threshold)
                    else:
                        DSC = Multi_Class_DSC(target.squeeze(1), output, out_channels, threshold, disregard_background)
                    # Multiply average DSC times the number of examples
                    valid_DSC += DSC * target.size(0)
                    pbar_val.set_postfix(loss=loss.item(), DSC=DSC)
                    # Release memory
                    del output, target, data, inputs, loss, DSC

                # Test loop
                if Test_Mask:
                    pbar_test = tqdm(test_loader, ncols=100, desc='Testing', leave=False)
                    for _, data in enumerate(pbar_test):
                        if use_custom_dataloader:
                            (inputs_t1c, inputs_t2f), target, _ = data
                            if train_on_gpu:
                                inputs_t1c = inputs_t1c.to('cuda', non_blocking=True)
                                inputs_t2f = inputs_t2f.to('cuda', non_blocking=True)
                                target = target.to('cuda', non_blocking=True)
                            inputs = (inputs_t1c, inputs_t2f)
                        else:
                            inputs, target, _ = data
                            if train_on_gpu:
                                inputs = inputs.to('cuda', non_blocking=True)
                                target = target.to('cuda', non_blocking=True)

                        # Forward pass
                        if use_custom_dataloader:
                            output = model(*inputs)
                        else:
                            output = model(inputs)

                        # Loss
                        if lossType == 'CrossEntropy':
                            loss = criterion(output, target.squeeze(1))
                        elif lossType == 'BCELoss':
                            output = torch.sigmoid(output)
                            loss = criterion(output.squeeze(1), target.float().squeeze(1))
                        elif lossType == 'DiceLoss':
                            output = torch.sigmoid(output)
                            loss = criterion(output, target)
                        elif lossType == 'CompoundLoss':
                            output = torch.sigmoid(output)
                            loss = criterion(output, target)
                        elif 'SMP' in lossType:
                            loss = criterion(output, target.squeeze(1))

                        # Multiply average loss times the number of examples in batch
                        test_loss += loss.item() * target.size(0)
                        # Calculate test DSC
                        if out_channels == 1:
                            DSC = DSC_training(target, output, threshold)
                        else:
                            DSC = Multi_Class_DSC(target.squeeze(1), output, out_channels, threshold, disregard_background)
                        # Multiply average DSC times the number of examples
                        test_DSC += DSC * target.size(0)
                        pbar_test.set_postfix(loss=loss.item(), DSC=DSC)
                        # Release memory
                        del output, target, data, inputs, loss, DSC

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                if Test_Mask:
                    test_loss = test_loss / len(test_loader.dataset)
                scheduler.step(valid_loss)

                # Calculate average DSC
                train_DSC = train_DSC / len(train_loader.dataset)
                valid_DSC = valid_DSC / len(valid_loader.dataset)
                if Test_Mask:
                    test_DSC = test_DSC / len(test_loader.dataset)
                    history.append([train_loss, valid_loss, test_loss, train_DSC, valid_DSC, test_DSC])
                else:
                    history.append([train_loss, valid_loss, train_DSC, valid_DSC])

                # Print training and validation results
                if Test_Mask:
                    if (epoch + 1) % print_every == 0:
                        print(f'\nEpoch: {epoch+1} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tTest Loss: {test_loss:.4f}')
                        print(f'\t\tTraining DSC: {100 * train_DSC:.2f}% \tValidation DSC: {100 * valid_DSC:.2f}% \tTest DSC: {100 * test_DSC:.2f}%')
                else:
                    if (epoch + 1) % print_every == 0:
                        print(f'\nEpoch: {epoch+1} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                        print(f'\t\tTraining DSC: {100 * train_DSC:.2f}% \tValidation DSC: {100 * valid_DSC:.2f}%')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Saving TrainModel
                    TrainChPoint = {}
                    TrainChPoint['model'] = model
                    TrainChPoint['history'] = history
                    torch.save(TrainChPoint, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_DSC = valid_DSC
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch+1}. Best epoch: {best_epoch+1} with loss: {valid_loss_min:.4f} and DSC: {100 * valid_best_DSC:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model = checkpoint['model']
                        # Attach the optimizer
                        model.optimizer = optimizer
                        if Test_Mask:
                            # Format history
                            history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'test_loss', 'train_DSC', 'val_DSC', 'test_DSC'])
                            return model, history
                        else:
                            # Format history
                            history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'train_DSC', 'val_DSC'])
                            return model, history

    # Load the best state dict
    checkpoint = torch.load(save_file_name)
    model = checkpoint['model']

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch+1} with loss: {valid_loss_min:.4f} and DSC: {100 * valid_best_DSC:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (n_epochs):.2f} seconds per epoch.')
    if Test_Mask:
        # Format history
        history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'test_loss', 'train_DSC', 'val_DSC', 'test_DSC'])
    else:
        # Format history
        history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'train_DSC', 'val_DSC'])
    return model, history