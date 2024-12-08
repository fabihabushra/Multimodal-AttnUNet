import torch
import torch.nn as nn

# Define helper functions and blocks used in the models
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,dropout_prob):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out,dropout_prob):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define the Attention U-Net model
# class AttU_Net(nn.Module):
#     def __init__(self, img_ch=3, output_ch=1):
#         super(AttU_Net, self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout_prob=0.3):
        super(AttU_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, dropout_prob=dropout_prob)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, dropout_prob=dropout_prob)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, dropout_prob=dropout_prob)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, dropout_prob=dropout_prob)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, dropout_prob=dropout_prob)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, dropout_prob=dropout_prob)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout_prob=dropout_prob)

        self.Up4 = up_conv(ch_in=512, ch_out=256, dropout_prob=dropout_prob)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout_prob=dropout_prob)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128, dropout_prob=dropout_prob)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout_prob=dropout_prob)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64, dropout_prob=dropout_prob)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout_prob=dropout_prob)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_intermediate=False):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if return_intermediate:
            encoder_features = [x1, x2, x3, x4, x5]
            decoder_features = [d2, d3, d4, d5]
            return encoder_features, decoder_features
        else:
            return d1

# Define the Fusion Attention U-Net model
# class FusionAttU_Net(nn.Module):
#     def __init__(self, img_ch=3, output_ch=1):
#         super(FusionAttU_Net, self).__init__()
        
#         # Initialize two AttU_Net models for each modality
#         self.unimodal_net1 = AttU_Net(img_ch, output_ch)
#         self.unimodal_net2 = AttU_Net(img_ch, output_ch)

#         # Define fusion branch layers with updated channel dimensions
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Encoder layers in the fusion branch
#         self.Conv1_fusion = conv_block(ch_in=128, ch_out=64)      # 64*2 channels -> 64 channels
#         self.Conv2_fusion = conv_block(ch_in=320, ch_out=128)     # 128*2 + 64 channels -> 128 channels
#         self.Conv3_fusion = conv_block(ch_in=640, ch_out=256)     # 256*2 + 128 channels -> 256 channels
#         self.Conv4_fusion = conv_block(ch_in=1280, ch_out=512)    # 512*2 + 256 channels -> 512 channels
#         self.Conv5_fusion = conv_block(ch_in=2560, ch_out=1024)   # 1024*2 + 512 channels -> 1024 channels
        
#         # Decoder layers in the fusion branch
#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Up_conv5_fusion = conv_block(ch_in=2048, ch_out=512)

#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv4_fusion = conv_block(ch_in=1024, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv3_fusion = conv_block(ch_in=512, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv2_fusion = conv_block(ch_in=256, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x1, x2):
#         # Pass each modality through the unimodal AttU_Nets, extracting both encoder and decoder features
#         x1_enc, x1_dec = self.unimodal_net1(x1, return_intermediate=True)
#         x2_enc, x2_dec = self.unimodal_net2(x2, return_intermediate=True)

#         # Encoding path of the fusion branch
#         f1_input = torch.cat((x1_enc[0], x2_enc[0]), dim=1)
#         #print(f"f1 input shape: {f1_input.shape}")
#         f1 = self.Conv1_fusion(f1_input)

#         f2_input = torch.cat((x1_enc[1], x2_enc[1], self.Maxpool(f1)), dim=1)
#         #print(f"f2 input shape: {f2_input.shape}")
#         f2 = self.Conv2_fusion(f2_input)

#         f3_input = torch.cat((x1_enc[2], x2_enc[2], self.Maxpool(f2)), dim=1)
#         #print(f"f3 input shape: {f3_input.shape}")
#         f3 = self.Conv3_fusion(f3_input)

#         f4_input = torch.cat((x1_enc[3], x2_enc[3], self.Maxpool(f3)), dim=1)
#         #print(f"f4 input shape: {f4_input.shape}")
#         f4 = self.Conv4_fusion(f4_input)

#         f5_input = torch.cat((x1_enc[4], x2_enc[4], self.Maxpool(f4)), dim=1)
#         #print(f"f5 input shape: {f5_input.shape}")
#         f5 = self.Conv5_fusion(f5_input)

#         # Decoding path of the fusion branch
#         d5_input = torch.cat((x1_dec[3], x2_dec[3], f4, self.Up5(f5)), dim=1)
#         #print(f"d5 input shape: {d5_input.shape}")
#         d5 = self.Up_conv5_fusion(d5_input)

#         d4_input = torch.cat((x1_dec[2], x2_dec[2], f3, self.Up4(d5)), dim=1)
#         #print(f"d4 input shape: {d4_input.shape}")
#         d4 = self.Up_conv4_fusion(d4_input)

#         d3_input = torch.cat((x1_dec[1], x2_dec[1], f2, self.Up3(d4)), dim=1)
#         #print(f"d3 input shape: {d3_input.shape}")
#         d3 = self.Up_conv3_fusion(d3_input)

#         d2_input = torch.cat((x1_dec[0], x2_dec[0], f1, self.Up2(d3)), dim=1)
#         #print(f"d2 input shape: {d2_input.shape}")
#         d2 = self.Up_conv2_fusion(d2_input)

#         d1 = self.Conv_1x1(d2)
#         return d1


class FusionAttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout_prob=0.3):
        super(FusionAttU_Net, self).__init__()
        
        # Initialize two AttU_Net models for each modality
        self.unimodal_net1 = AttU_Net(img_ch, output_ch, dropout_prob=dropout_prob)
        self.unimodal_net2 = AttU_Net(img_ch, output_ch, dropout_prob=dropout_prob)

        # Define fusion branch layers with updated channel dimensions and dropout
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder layers in the fusion branch
        self.Conv1_fusion = conv_block(ch_in=128, ch_out=64, dropout_prob=dropout_prob)
        self.Conv2_fusion = conv_block(ch_in=320, ch_out=128, dropout_prob=dropout_prob)
        self.Conv3_fusion = conv_block(ch_in=640, ch_out=256, dropout_prob=dropout_prob)
        self.Conv4_fusion = conv_block(ch_in=1280, ch_out=512, dropout_prob=dropout_prob)
        self.Conv5_fusion = conv_block(ch_in=2560, ch_out=1024, dropout_prob=dropout_prob)
        
        # Decoder layers in the fusion branch
        self.Up5 = up_conv(ch_in=1024, ch_out=512, dropout_prob=dropout_prob)
        self.Up_conv5_fusion = conv_block(ch_in=2048, ch_out=512, dropout_prob=dropout_prob)

        self.Up4 = up_conv(ch_in=512, ch_out=256, dropout_prob=dropout_prob)
        self.Up_conv4_fusion = conv_block(ch_in=1024, ch_out=256, dropout_prob=dropout_prob)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128, dropout_prob=dropout_prob)
        self.Up_conv3_fusion = conv_block(ch_in=512, ch_out=128, dropout_prob=dropout_prob)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64, dropout_prob=dropout_prob)
        self.Up_conv2_fusion = conv_block(ch_in=256, ch_out=64, dropout_prob=dropout_prob)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x1, x2):
        # Pass each modality through the unimodal AttU_Nets, extracting both encoder and decoder features
        x1_enc, x1_dec = self.unimodal_net1(x1, return_intermediate=True)
        x2_enc, x2_dec = self.unimodal_net2(x2, return_intermediate=True)

        # Encoding path of the fusion branch
        f1_input = torch.cat((x1_enc[0], x2_enc[0]), dim=1)
        f1 = self.Conv1_fusion(f1_input)

        f2_input = torch.cat((x1_enc[1], x2_enc[1], self.Maxpool(f1)), dim=1)
        f2 = self.Conv2_fusion(f2_input)

        f3_input = torch.cat((x1_enc[2], x2_enc[2], self.Maxpool(f2)), dim=1)
        f3 = self.Conv3_fusion(f3_input)

        f4_input = torch.cat((x1_enc[3], x2_enc[3], self.Maxpool(f3)), dim=1)
        f4 = self.Conv4_fusion(f4_input)

        f5_input = torch.cat((x1_enc[4], x2_enc[4], self.Maxpool(f4)), dim=1)
        f5 = self.Conv5_fusion(f5_input)

        # Decoding path of the fusion branch
        d5_input = torch.cat((x1_dec[3], x2_dec[3], f4, self.Up5(f5)), dim=1)
        d5 = self.Up_conv5_fusion(d5_input)

        d4_input = torch.cat((x1_dec[2], x2_dec[2], f3, self.Up4(d5)), dim=1)
        d4 = self.Up_conv4_fusion(d4_input)

        d3_input = torch.cat((x1_dec[1], x2_dec[1], f2, self.Up3(d4)), dim=1)
        d3 = self.Up_conv3_fusion(d3_input)

        d2_input = torch.cat((x1_dec[0], x2_dec[0], f1, self.Up2(d3)), dim=1)
        d2 = self.Up_conv2_fusion(d2_input)

        d1 = self.Conv_1x1(d2)
        return d1


# Testing the model with a dummy input
batch_size = 8
dummy_input1 = torch.randn(batch_size, 3, 256, 256)  # First modality
dummy_input2 = torch.randn(batch_size, 3, 256, 256)  # Second modality

model = FusionAttU_Net(img_ch=3, output_ch=1)
output = model(dummy_input1, dummy_input2)
#print("Output shape:", output.shape)
