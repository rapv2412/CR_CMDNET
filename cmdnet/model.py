import torch 
import torch.nn as nn
import torch.nn.functional as F

class Network33(nn.Module):
        
        # In channels 4, represent 4 bands (low res RGB, 1 SAR)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # Mapping Layer
        self.map = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3)
        # Out put high res RGB

    def forward(self,t):

        t = F.relu(self.conv1(t))  
        t = F.relu(self.conv2(t))   
        t = F.relu(self.conv3(t))                      
        t = F.relu(self.conv4(t))   
        t = F.relu(self.map(t))   
        t = F.relu(self.deconv1(t))   
        t = F.relu(self.deconv2(t))     
        t = F.relu(self.deconv3(t))   
        t = F.relu(self.deconv4(t))

# Same as above but slight modified architecture
class Network55(nn.Module):

    def __init__(self):
        super(Network,self).__init__()

        # 5x5 kernsl size instead of 3x3, only need two encoding and decoding layers for similar field of view
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)

        self.map = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5)



    def forward(self,t):

        t = F.relu(self.conv1(t))  
        t = F.relu(self.conv2(t))     
        t = F.relu(self.map(t))      
        t = F.relu(self.deconv3(t))   
        t = F.relu(self.deconv4(t))

        return t  

def norm_replace(imcloudy, impred, imrepl == False):

    sizexy = imcloudy[0][0].size() #Get size
    sizex, sizey = sizexy[0] , sizexy[1]

    imnorm = impred.clone()
    imrepl = imcloudy.clone()

    for i in range(len(imcloudy)):
        for j in range(len(imcloudy[0])):

            imc = imcloudy[i][j]
            imp = impred[i][j]
            
            #Assume Cloudy pixels have value = 1
            #Added !=1 to exclude cloudy pixels from mean/std calcs

            #Pf is from cloud contaminated image
            Pf_mu = imc[imc!=1].mean() 
            Pf_sig = imc[imc!=1].std()

            #Pe is from estimated/predicted image
            Pe_mu = imp.mean() 
            Pe_sig = imp.std()

            #only do normalization 
            for x in range(sizex):
                for y in range(sizey):
                    imnorm[i][j][x][y] = (imp[x][y] - Pe_mu)*(Pf_sig/Pe_sig)+Pf_mu
                    if imc[x][y] == 1 & imrepl == True :
                        imrepl[i][j][x][y] = imnorm[i][j][x][y]

    if imrepl == False:
        return imnorm
    else:
        return imnorm, imrepl
