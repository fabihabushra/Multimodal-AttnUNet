# CNN configuration file

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######
config['parentdir'] = r'D:\Rusab/'            # main directory
config['ONN'] = False                                                                     # set to 'True' if you are using ONN
config['batch_size'] = 8                                                                # batch size, Change to fit hardware
config['in_channels'] = 1                                                                # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays
config['out_channels'] = 1                                                               # number of classes (Use 1 for binary, Use total number of classes with background for multiclass)
config['palette'] = [0,0,0,71,39,119,62,73,137,48,103,141,37,130,142,30,157,136,53,183,120,109,206,88,181,221,43,253,231,36]
config['disregard_background'] = True                                                    # For Multi-Class Only : Whether to Disregard Background (0 Class) in metric calculation
config['calculate_loss_on_bg']  = False                                                  # For Multi-Class Only : Whether to calculate loss on background class
config['input_mean'] = [0.0577]                                             # provide 3 numbers for RGB images or 1 number for gray scale images in list format
config['input_std'] = [0.1413]                                             # provide 3 numbers for RGB images or 1 number for gray scale images in list format
config['input_mean_t1c'] = [0.0567]
config['input_std_t1c'] = [0.1395]
config['input_mean_t2f'] = [0.0878]
config['input_std_t2f'] = [0.211]
config['optim_fc'] = 'Adam'                                                              # 'Adam' or 'SGD'
config['lr'] = 5e-5                                                                      # learning rate
config['class_weights'] = None                                                           # class weights for multi class masks, default: none
config['lossType'] = '0.5*SMP_DiceLoss+0.5*SMP_JaccardLoss'                                          # loss function: 'CrossEntropy' for multi-class. 'BCELoss' or 'DiceLoss' for binary class
config['n_epochs']  = 100                                                               # number of training epochs
config['epochs_patience'] = 6                                                            # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
config['lr_factor'] = 0.15
config['max_epochs_stop'] = 20                                                            # maximum number of epochs with no improvement in validation loss for early stopping
config['num_folds']  = 5                                                                 # number of cross validation folds
config['Resize_h'] = 240                                                                 # network input size
config['Resize_w'] = config['Resize_h']
# config['load_model'] = config['parentdir'] + 'load_model/Densenet121_FPN_lung_seg_fold_1.pt'    # specify path of pretrained model wieghts or set to False to train from scratch
config['load_model'] = False                                                             # specify path of pretrained model wieghts or set to False to train from scratch
config['Test_Mask'] = True                                                               # set to true if you have the test masks, to compute Evaluation metricies over test set
config['model_type'] = 'Custom'                                                             # SMP libary models : SMP | ONN-Based Decoders : ONN_dec | Custom models : Custom
config['model_to_load'] = 'FusionAttU_Net'                                     # enter model name like this: <encoder_name>*<decoder_name> (names should match the documentation exactly)
config['model_name'] = 'WT_FusionAttU_Net_Multimodal_sample(2)_DJ_dropout'                              # choose a unique name for result folder
config['decoder_attention'] = None                                                  # Turn on Attention Layer in Unet/Unet++ (None / 'scse') (only works with SMP and ONN_dec)
config['encoder_depth']  = 5                                                            # number of encoder layers (For pretrained weights default: 5) (not usable with PAN decoder)
config['encoder_weights'] = 'imagenet'                                                  # pretrained weights: 'imagenet' | Train from scratch: None (some encoders have multiple weights check documentation)
config['activation'] = None                                                             # last layer activation function (default: None | Available functions: provided in the list above)
config['q_order'] = 3                                                                    # ONN q-order
config['max_shift'] = 0                                                                  # ONN max_shift
config['seg_threshold'] = 0.5                                                           # Segmentation Threshold (Default 0.5)
config['U_init_features'] = 32                                                           # Only for model_type : Custom | number of kernals in the first UNet conv layer ('32' & '64' are common values)
config['unfolding_decay'] = 2                                                            # Only for model_type : Custom : CSC & CSCA models | Default Values: Unfolding = 1 for CSC Decay = 2 for CSCA
config['fold_to_run'] = [4,4] # define as [] to loop through all folds, or specify start and end folds i.e. [3, 5] or [5, 5]
config['Results_path'] = r"D:\Rusab/" + "Results"   # main results file
config['save_path'] = config['Results_path'] +'/'+ config['model_name']                  # save path
config['generated_masks']  =  r'D:\Rusab/' + 'Generated_mask'         # path to save generated_masks for test set
##################
