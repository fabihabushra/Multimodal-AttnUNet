# TrainUnet.py

# PyTorch and Torchvision
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import save_image

# Data science tools
import numpy as np
import pandas as pd
import os
from skimage import io
from PIL import Image

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Loss functions and metrics
from segmentation_models_pytorch.losses import (
    DiceLoss as smp_DiceLoss, JaccardLoss, FocalLoss, LovaszLoss,
    SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, MCCLoss, TverskyLoss
)
import segmentation_models_pytorch as smp

# Utilities and models
from os import path
from utils import *
from models import get_pretrained_model
from importlib import import_module
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Import the custom dataset for dual modality
from custom_dataset import DualModalityDataset  # You need to have this file as per previous instructions

import tracemalloc
tracemalloc.start()

# Define the combined loss function if necessary
class SMP_CombinedLoss(torch.nn.Module):
    def __init__(self, loss_types, weights, smp_loss_mode):
        super(SMP_CombinedLoss, self).__init__()
        print("Combined Loss initiated with:")
        print(f"loss_types: {loss_types}, weights: {weights}")
        self.losses = []
        for i, lossType in enumerate(loss_types):
            if lossType == 'SMP_DiceLoss':
                self.losses.append((smp_DiceLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_JaccardLoss':
                self.losses.append((JaccardLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_FocalLoss':
                self.losses.append((FocalLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_LovaszLoss':
                self.losses.append((LovaszLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_SoftBCEWithLogitsLoss':
                self.losses.append((SoftBCEWithLogitsLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_TverskyLoss':
                self.losses.append((TverskyLoss(mode=smp_loss_mode, alpha=0.3, beta=0.7), weights[i]))
            elif lossType == 'SMP_SoftCrossEntropyLoss':
                self.losses.append((SoftCrossEntropyLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_MCCLoss':
                self.losses.append((MCCLoss(mode=smp_loss_mode), weights[i]))

    def forward(self, inputs, targets):
        combined_loss = 0
        for loss, weight in self.losses:
            combined_loss += weight * loss(inputs, targets)
        return combined_loss

# Metrics functions
def Multi_Class_DSC(target, output, C, threshold, disregard_background=True):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes=C)
    DSC = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        DSC = DSC[:, 1:]
    return DSC.mean(dim=0)

def Multi_Class_IoU(target, output, C, threshold, disregard_background=True):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes=C)
    IOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        IOU = IOU[:, 1:]
    return IOU.mean(dim=0)

# Parse command line arguments
fname = "config.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config

if __name__ == '__main__':
    ################## Network hyper-parameters
    parentdir = config['parentdir']                     # main directory
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    in_channels = config['in_channels']                 # 1 for gray scale images
    out_channels = config['out_channels']               # '1' for binary class losses, 'number of classes' for multi-class losses
    palette = config['palette']
    disregard_background = config['disregard_background']
    # Mean and std for t1c modality
    input_mean_t1c = config['input_mean_t1c']           # Dataset mean for t1c modality
    input_std_t1c = config['input_std_t1c']             # Dataset std for t1c modality
    # Mean and std for t2f modality
    input_mean_t2f = config['input_mean_t2f']           # Dataset mean for t2f modality
    input_std_t2f = config['input_std_t2f']             # Dataset std for t2f modality

    optim_fc = config['optim_fc']                       # 'Adam' or 'SGD'
    lr = config['lr']                                   # learning rate
    if 'calculate_loss_on_bg' in config:
        calculate_loss_on_bg = config['calculate_loss_on_bg']  # set to True if you are using multi-label masks
    else:
        calculate_loss_on_bg = True
    class_weights = config['class_weights']             # class weights for multi class masks, default: none
    lossType = config['lossType']                       # loss function
    n_epochs = config['n_epochs']                       # number of training epochs
    epochs_patience = config['epochs_patience']         # patience for LR scheduler
    lr_factor = config['lr_factor']                     # factor for LR scheduler
    max_epochs_stop = config['max_epochs_stop']         # early stopping
    num_folds = config['num_folds']                     # number of cross validation folds
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']
    load_model = config['load_model']                   # specify path of pretrained model weights or False
    Test_Mask = config['Test_Mask']                     # set to true if you have the test masks
    model_type = config['model_type']                   # SMP library models : SMP, Custom models : Custom
    model_to_load = config['model_to_load']             # e.g., 'UNet' or 'FusionAttU_Net'
    model_name = config['model_name']                   # name of result folder
    decoder_attention = config['decoder_attention']     # attention layer
    encoder_depth = config['encoder_depth']             # number of encoder layers
    encoder_weights = config['encoder_weights']         # pretrained weights or None
    activation = config['activation']                   # last layer activation function
    q_order = config['q_order']                         # ONN q-order
    max_shift = config['max_shift']                     # ONN max-shift
    seg_threshold = config['seg_threshold']             # Segmentation Threshold (Default 0.5)
    U_init_features = config['U_init_features']         # number of kernels in the first UNet conv layer
    if 'unfolding_decay' in config:
        unfolding_decay = config['unfolding_decay']     # unfolding or decay in the CSC_UNet or CSCA_UNet
    else:
        unfolding_decay = 1
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path
    generated_masks = config['generated_masks']         # path to save generated masks for test set
    ##################
    traindir = parentdir + 'Data/Train/'
    testdir = parentdir + 'Data/Test/'
    valdir = parentdir + 'Data/Val/'
    # Create Directory
    os.makedirs(Results_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(generated_masks, exist_ok=True)
    generated_masks = os.path.join(generated_masks, model_name)
    os.makedirs(generated_masks, exist_ok=True)

    shutil.copy('config.py', save_path)

    # Whether to train on a GPU
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    # Number of GPUs
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPUs detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    test_history = []
    index = []
    # Loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds + 1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1] + 1

    # Determine if we're using the FusionAttU_Net model
    use_custom_dataloader = model_to_load == 'FusionAttU_Net'

    for fold_idx in range(loop_start, loop_end):
        print('#############################################################')
        if fold_idx == loop_start:
            print('Training using ' + model_to_load + ' network')
        print(f'Started fold {fold_idx}')
        save_file_name = os.path.join(save_path, f'{model_name}_fold_{fold_idx}.pt')
        checkpoint_name = os.path.join(save_path, 'checkpoint.pt')
        traindir_fold = os.path.join(traindir, f'fold_{fold_idx}/')
        testdir_fold = os.path.join(testdir, f'fold_{fold_idx}/')
        valdir_fold = os.path.join(valdir, f'fold_{fold_idx}/')

        gen_fold_mask = os.path.join(generated_masks, f'fold_{fold_idx}')
        os.makedirs(gen_fold_mask, exist_ok=True)

        # Create train labels
        categories, n_Class_train, img_names_train = Createlabels(traindir_fold, Seg_state=True)
        class_num = len(categories)
        # Create val labels
        _, n_Class_val, img_names_val = Createlabels(valdir_fold, Seg_state=True)
        # Create test labels
        _, n_Class_test, img_names_test = Createlabels(testdir_fold, Seg_state=True)

        # Random shuffle before training
        np.random.shuffle(img_names_train)

        # Create datasets and dataloaders
        if use_custom_dataloader:
            # Use the custom DualModalityDataset
            train_ds = DualModalityDataset(
                root_dir=traindir_fold,
                images_t1c_dir='images_t1c',
                images_t2f_dir='images_t2f',
                masks_dir='masks',
                img_names=img_names_train,
                h=Resize_h,
                w=Resize_w,
                mean_t1c=input_mean_t1c,
                std_t1c=input_std_t1c,
                mean_t2f=input_mean_t2f,
                std_t2f=input_std_t2f,
                return_path=False
            )
            val_ds = DualModalityDataset(
                root_dir=valdir_fold,
                images_t1c_dir='images_t1c',
                images_t2f_dir='images_t2f',
                masks_dir='masks',
                img_names=img_names_val,
                h=Resize_h,
                w=Resize_w,
                mean_t1c=input_mean_t1c,
                std_t1c=input_std_t1c,
                mean_t2f=input_mean_t2f,
                std_t2f=input_std_t2f,
                return_path=False
            )
            test_ds = DualModalityDataset(
                root_dir=testdir_fold,
                images_t1c_dir='images_t1c',
                images_t2f_dir='images_t2f',
                masks_dir='masks' if Test_Mask else None,
                img_names=img_names_test,
                h=Resize_h,
                w=Resize_w,
                mean_t1c=input_mean_t1c,
                std_t1c=input_std_t1c,
                mean_t2f=input_mean_t2f,
                std_t2f=input_std_t2f,
                return_path=True
            )
        else:
            # Use the default SegData or TestData classes
            train_ds = SegData(
                root_dir=traindir_fold,
                images_path='images',
                masks_path='masks',
                img_names=img_names_train,
                h=Resize_h,
                w=Resize_w,
                mean=input_mean_t1c,  # Use input_mean_t1c or adjust accordingly
                std=input_std_t1c,    # Use input_std_t1c or adjust accordingly
                in_channels=in_channels,
                out_channels=out_channels,
                return_path=False,
                ONN=ONN
            )
            val_ds = SegData(
                root_dir=valdir_fold,
                images_path='images',
                masks_path='masks',
                img_names=img_names_val,
                h=Resize_h,
                w=Resize_w,
                mean=input_mean_t1c,  # Use input_mean_t1c or adjust accordingly
                std=input_std_t1c,    # Use input_std_t1c or adjust accordingly
                in_channels=in_channels,
                out_channels=out_channels,
                return_path=False,
                ONN=ONN
            )
            if Test_Mask:
                test_ds = SegData(
                    root_dir=testdir_fold,
                    images_path='images',
                    masks_path='masks',
                    img_names=img_names_test,
                    h=Resize_h,
                    w=Resize_w,
                    mean=input_mean_t1c,  # Use input_mean_t1c or adjust accordingly
                    std=input_std_t1c,    # Use input_std_t1c or adjust accordingly
                    in_channels=in_channels,
                    out_channels=out_channels,
                    return_path=True,
                    ONN=ONN
                )
            else:
                test_ds = TestData(
                    root_dir=testdir_fold,
                    images_path='images',
                    img_names=img_names_test,
                    h=Resize_h,
                    w=Resize_w,
                    mean=input_mean_t1c,  # Use input_mean_t1c or adjust accordingly
                    std=input_std_t1c,    # Use input_std_t1c or adjust accordingly
                    in_channels=in_channels,
                    return_path=True,
                    ONN=ONN
                )

        # Create DataLoaders
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=len(train_ds) % batch_size == 1
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        # Release memory
        del n_Class_train, img_names_train
        del n_Class_val, img_names_val
        del n_Class_test, img_names_test
        torch.cuda.empty_cache()

        if out_channels == 1:
            smp_loss_mode = 'binary'
        else:
            smp_loss_mode = 'multiclass'

        # Load model
        if load_model:
            checkpoint = torch.load(load_model)
            model = checkpoint['model']
            history = checkpoint['history']
            start_epoch = len(history)
            print('Resuming training from checkpoint\n')
            del checkpoint
        else:
            history = []
            start_epoch = 0
            model = get_pretrained_model(
                parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift,
                U_init_features, in_channels, out_channels, train_on_gpu, multi_gpu, decoder_attention,
                activation, unfolding_decay
            )
            model = model.to('cuda')

        # Check if model is on CUDA
        if next(model.parameters()).is_cuda:
            print('Model device: cuda')

        # Choose loss function and optimizer
        combined_loss_flag = False
        if '+' in lossType:
            lossType = lossType.replace(' ', '')
            lossType_list = lossType.split('+')
            weights_list = []
            losses_list = []
            for smp_loss in lossType_list:
                weights_list.append(float(smp_loss.split('*')[0]))
                losses_list.append(smp_loss.split('*')[1])

            criterion = SMP_CombinedLoss(losses_list, weights_list, smp_loss_mode)
            combined_loss_flag = True

        else:
            if disregard_background and calculate_loss_on_bg and out_channels != 1:
                class_list = list(range(1, out_channels))
            else:
                class_list = None
            if lossType == 'CrossEntropy':
                if class_weights is not None:
                    class_weights = torch.tensor(class_weights).cuda()
                criterion = nn.CrossEntropyLoss(class_weights)
            elif lossType == 'BCELoss':
                criterion = nn.BCELoss()
            elif lossType == 'DiceLoss':
                from unet_loss import DiceLoss
                criterion = DiceLoss()
            elif lossType == 'CompoundLoss':
                from unet_loss import CompoundLoss
                criterion = CompoundLoss()
            elif lossType == 'SMP_DiceLoss':
                criterion = smp_DiceLoss(mode=smp_loss_mode, classes=class_list)
            elif lossType == 'SMP_JaccardLoss':
                criterion = JaccardLoss(mode=smp_loss_mode, classes=class_list)
            elif lossType == 'SMP_FocalLoss':
                criterion = FocalLoss(mode=smp_loss_mode)
            elif lossType == 'SMP_LovaszLoss':
                criterion = LovaszLoss(mode=smp_loss_mode)
            elif lossType == 'SMP_SoftBCEWithLogitsLoss':
                criterion = SoftBCEWithLogitsLoss(mode=smp_loss_mode)
            elif lossType == 'SMP_TverskyLoss':
                criterion = TverskyLoss(mode=smp_loss_mode, classes=class_list, alpha=0.3, beta=0.7)
            elif lossType == 'SMP_SoftCrossEntropyLoss':
                criterion = SoftCrossEntropyLoss(mode=smp_loss_mode)
            elif lossType == 'SMP_MCCLoss':
                criterion = MCCLoss(mode=smp_loss_mode)

        # Optimizer
        if optim_fc == 'Adam':
            optimizer = optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                weight_decay=0.0001, amsgrad=False
            )
        elif optim_fc == 'SGD':
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, dampening=0,
                weight_decay=0.0001, nesterov=False
            )
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=False,
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
        )

        trainable_params = sum(p.numel() for p in model.parameters(recurse=True) if p.requires_grad) / 1e6
        print(f'Trainable parameters: {trainable_params} million')

        if combined_loss_flag:
            # This is done to keep if SMP condition in utils.py for losses (lazy coding)
            lossType_temp = lossType
            lossType = lossType_list[0]

        # Training loop
        model, history = train(
            out_channels,
            model,
            criterion,
            optimizer,
            lossType,
            scheduler,
            train_dl,
            val_dl,
            test_dl,
            Test_Mask,
            seg_threshold,
            checkpoint_name,
            train_on_gpu,
            history=history,
            max_epochs_stop=max_epochs_stop,
            start_epoch=start_epoch,
            n_epochs=n_epochs,
            print_every=1,
            disregard_background=disregard_background,
            use_custom_dataloader=use_custom_dataloader  # Pass this to the train function
        )

        # Save the trained model
        TrainChPoint = {}
        TrainChPoint['model'] = model
        TrainChPoint['history'] = history
        torch.save(TrainChPoint, save_file_name)

        # Training Results
        if Test_Mask:
            # Plot loss
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss', 'test_loss']:
                plt.plot(history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(save_path, f'LossPerEpoch_fold_{fold_idx}.png'))

            # Plot accuracy
            plt.figure(figsize=(8, 6))
            for c in ['train_DSC', 'val_DSC', 'test_DSC']:
                plt.plot(100 * history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('DSC')
            plt.savefig(os.path.join(save_path, f'DSCPerEpoch_fold_{fold_idx}.png'))
        else:
            # Plot loss
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss']:
                plt.plot(history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(save_path, f'LossPerEpoch_fold_{fold_idx}.png'))

            # Plot accuracy
            plt.figure(figsize=(8, 6))
            for c in ['train_DSC', 'val_DSC']:
                plt.plot(100 * history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('DSC')
            plt.savefig(os.path.join(save_path, f'DSCPerEpoch_fold_{fold_idx}.png'))

        # Release memory
        del optimizer, TrainChPoint, scheduler
        del train_ds, train_dl, val_ds, val_dl
        torch.cuda.empty_cache()

        # Conversion factor for masks
        if out_channels == 1:
            conv_fact = 255.0 / 1.0
        else:
            conv_fact = 255.0 / (out_channels - 1)

        # Evaluation on test set
        model.eval()
        if Test_Mask:
            test_acc = 0.0
            test_loss = 0.0
            test_IoU = 0.0
            test_DSC = 0.0
            for data in test_dl:
                if use_custom_dataloader:
                    (inputs_t1c, inputs_t2f), target, im_path = data
                    inputs_t1c = inputs_t1c.to('cuda', non_blocking=True)
                    inputs_t2f = inputs_t2f.to('cuda', non_blocking=True)
                    inputs = (inputs_t1c, inputs_t2f)
                else:
                    data, target, im_path = data
                    inputs = data.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)

                # Compute network output
                with torch.no_grad():
                    output = model(*inputs) if use_custom_dataloader else model(inputs)
                # Compute loss
                if lossType == 'CrossEntropy':
                    loss = criterion(output, target.squeeze(1))
                    output_temp = F.softmax(output, dim=1)
                    _, output = torch.max(output_temp, dim=1)
                elif lossType == 'BCELoss':
                    output = torch.sigmoid(output)
                    loss = criterion(output.squeeze(1), target.float().squeeze(1))
                    output = 1.0 * (output > seg_threshold)
                elif lossType == 'DiceLoss':
                    output = torch.sigmoid(output)
                    loss = criterion(output, target)
                    output = 1.0 * (output > seg_threshold)
                elif lossType == 'CompoundLoss':
                    output = torch.sigmoid(output)
                    loss = criterion(output, target)
                    output = 1.0 * (output > seg_threshold)
                elif 'SMP' in lossType:
                    loss = criterion(output, target.squeeze(1))
                    if out_channels == 1:
                        output = torch.sigmoid(output)
                        output = 1.0 * (output > seg_threshold)
                    else:
                        output = F.softmax(output, dim=1)
                        _, output = torch.max(output, dim=1)

                test_loss += loss.item() * inputs_t1c.size(0)
                # Compute accuracy
                accuracy = calc_acc(target, output, seg_threshold)
                test_acc += accuracy.item() * inputs_t1c.size(0)
                if out_channels == 1:
                    # Compute IoU
                    IoU = compute_IoU(target, output, seg_threshold)
                    test_IoU += IoU * inputs_t1c.size(0)
                    # Compute DSC
                    DSC = compute_DSC(target, output, seg_threshold)
                    test_DSC += DSC * inputs_t1c.size(0)
                else:
                    # Compute IoU and DSC for multiclass
                    IoU = Multi_Class_IoU(target.squeeze(1), output, out_channels, seg_threshold, disregard_background)
                    test_IoU += IoU * inputs_t1c.size(0)
                    DSC = Multi_Class_DSC(target.squeeze(1), output, out_channels, seg_threshold, disregard_background)
                    test_DSC += DSC * inputs_t1c.size(0)

                # Write generated masks to file
                for i in range(inputs_t1c.shape[0]):
                    tensor_image = output[i].squeeze(0).cpu()
                    image_name = im_path[i]
                    filename = os.path.join(gen_fold_mask, image_name)
                    tensor_image = tensor_image.numpy()
                    if out_channels == 1:
                        tensor_image = conv_fact * tensor_image
                        tensor_image = tensor_image.astype('uint8')
                        io.imsave(filename, tensor_image, check_contrast=False)
                    else:
                        tensor_image = tensor_image.astype('uint8')
                        pil_image = Image.fromarray(tensor_image)
                        pil_image = pil_image.convert('P')
                        pil_image.putpalette(palette)
                        pil_image.save(filename)
        else:
            for data in test_dl:
                if use_custom_dataloader:
                    (inputs_t1c, inputs_t2f), im_path = data
                    inputs_t1c = inputs_t1c.to('cuda', non_blocking=True)
                    inputs_t2f = inputs_t2f.to('cuda', non_blocking=True)
                    inputs = (inputs_t1c, inputs_t2f)
                else:
                    data, im_path = data
                    inputs = data.to('cuda', non_blocking=True)

                # Compute network output
                with torch.no_grad():
                    output = model(*inputs) if use_custom_dataloader else model(inputs)

                # Process output
                if lossType == 'CrossEntropy':
                    output = F.softmax(output, dim=1)
                    _, output = torch.max(output, dim=1)
                elif lossType == 'BCELoss':
                    output = torch.sigmoid(output)
                    output = 1.0 * (output > seg_threshold)
                elif lossType == 'DiceLoss':
                    output = torch.sigmoid(output)
                    output = 1.0 * (output > seg_threshold)
                elif lossType == 'CompoundLoss':
                    output = torch.sigmoid(output)
                    output = 1.0 * (output > seg_threshold)
                elif 'SMP' in lossType:
                    if out_channels == 1:
                        output = torch.sigmoid(output)
                        output = 1.0 * (output > seg_threshold)
                    else:
                        output = F.softmax(output, dim=1)
                        _, output = torch.max(output, dim=1)

                # Write generated masks to file
                for i in range(inputs_t1c.shape[0]):
                    tensor_image = output[i].squeeze(0).cpu()
                    image_name = im_path[i]
                    filename = os.path.join(gen_fold_mask, image_name)
                    tensor_image = tensor_image.numpy()
                    if out_channels == 1:
                        tensor_image = conv_fact * tensor_image
                        tensor_image = tensor_image.astype('uint8')
                        io.imsave(filename, tensor_image, check_contrast=False)
                    else:
                        tensor_image = tensor_image.astype('uint8')
                        pil_image = Image.fromarray(tensor_image)
                        pil_image = pil_image.convert('P')
                        pil_image.putpalette(palette)
                        pil_image.save(filename)

        if Test_Mask:
            test_loss = test_loss / len(test_dl.dataset)
            test_loss = round(test_loss, 4)
            test_acc = test_acc / len(test_dl.dataset)
            test_acc = round(100 * test_acc, 2)
            test_IoU = test_IoU / len(test_dl.dataset)
            if out_channels != 1:
                test_IoU_per_class = test_IoU
                test_IoU = test_IoU.mean().item()
            test_IoU = round(100 * test_IoU, 2)
            test_DSC = test_DSC / len(test_dl.dataset)
            if out_channels != 1:
                test_DSC_per_class = test_DSC
                test_DSC = test_DSC.mean().item()
            test_DSC = round(100 * test_DSC, 2)
            fold_test_history = [test_loss, test_acc, test_IoU, test_DSC]
            if out_channels != 1:
                if disregard_background:
                    class_num = out_channels - 1
                else:
                    class_num = out_channels

                for clas_i in range(0, class_num):
                    fold_test_history.append(round(100 * test_IoU_per_class[clas_i].item(), 2))
                    fold_test_history.append(round(100 * test_DSC_per_class[clas_i].item(), 2))
            test_history.append(fold_test_history)
            index.extend([f'fold_{fold_idx}'])
            print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc:.2f}%')
            print(f'Test IoU: {test_IoU},  Test DSC: {test_DSC}')
            del target, loss, test_loss, accuracy, test_acc
            del IoU, test_IoU, DSC, test_DSC

        # Release memory
        del model, criterion, history, test_ds, test_dl
        del data, output
        torch.cuda.empty_cache()

        print(f'Completed fold {fold_idx}')

        # Store lossType back for the new fold
        if combined_loss_flag:
            lossType = lossType_temp

    print('#############################################################')

    if os.path.exists(checkpoint_name):
        os.remove(checkpoint_name)
        print("Checkpoint File Removed!")

    if Test_Mask:
        # Saving Test Results
        TestChPoint = {}
        TestChPoint['test_history'] = test_history
        columns_names = ['loss', 'Accuracy', 'IoU', 'DSC']
        if disregard_background:
            first_class = 1
        else:
            first_class = 0
        for clas_i in range(first_class, out_channels):
            columns_names.append(f'IoU_{clas_i}')
            columns_names.append(f'DSC_{clas_i}')
        temp = pd.DataFrame(test_history, columns=columns_names)
        # Compute average values
        test_loss = np.mean(temp['loss'])
        test_acc = np.mean(temp['Accuracy'])
        test_IoU = np.mean(temp['IoU'])
        test_DSC = np.mean(temp['DSC'])
        avg_test_history = [test_loss, test_acc, test_IoU, test_DSC]
        N = 2 * (out_channels - 1)
        for i in range(4, 4 + N):
            current_value = np.mean(temp[columns_names[i]])
            avg_test_history.append(current_value)
        test_history.append(avg_test_history)
        index.extend(['Average'])
        test_history = pd.DataFrame(test_history, columns=columns_names, index=index)
        save_file = os.path.join(save_path, model_name + '_test_results.pt')
        torch.save(TestChPoint, save_file)
        # Save to Excel file
        save_file = os.path.join(save_path, model_name + '.xlsx')
        writer = pd.ExcelWriter(save_file, engine='openpyxl')
        col = 2
        row = 2
        test_history.to_excel(writer, "Results", startcol=col, startrow=row)
        writer._save()

        print('\n')
        print(test_history)
        print('\n')

        print('#############################################################')
