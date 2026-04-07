import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):

    def __init__(self, in_channels=9, out_channels=3):
        super(TrackNet,self).__init__

        self.conv1_1 = nn.Conv2(in_channels,64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # --- MÓDULO DECODIFICADOR (DeconvNet) ---
        # Uso iterativo de interpolación bilineal o nearest para evitar artefactos en cuadrícula
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.deconv1_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.deconv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv1_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.deconv2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deconv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.deconv3_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # --- PROYECCIÓN DE SALIDA ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def fordward(self,x):

         # Propagación a través del codificador con activaciones no lineales
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        
        # Propagación a través del decodificador restaurando resolución
        x = self.upsample1(x)
        x = F.relu(self.deconv1_1(x))
        x = F.relu(self.deconv1_2(x))
        x = F.relu(self.deconv1_3(x))
        
        x = self.upsample2(x)
        x = F.relu(self.deconv2_1(x))
        x = F.relu(self.deconv2_2(x))

        x = self.upsample3(x)
        x = F.relu(self.deconv3_1(x))
        x = F.relu(self.deconv3_2(x))
        
        # Transformación a mapa de probabilidad acotado en 
        out = self.final_conv(x)
        out = self.sigmoid(out)
        
        return out
