import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from einops.layers.torch import Rearrange
import xarray as xr
from PIL import Image
import shutil
import imageio


def main():
    return

###########################################################################################################################################################################
#helper functions


#loader to get datm/gatm address & return allNoise, allSignalNoise, allResidues np arrays
def DATM_GATM_loader_NParrays(DATM_GATM_address, imgDimensions):    #example input DEFAULT_DATM_GATM_address, imgDimensions = [x,y,channel=1(greyscale)]
    folderNoise = DATM_GATM_address + 'gatm/'  ##noise only = gatm
    folderSignalandNoise = DATM_GATM_address + 'datm/' ##signal+noise = datm
    filesSN = os.listdir(folderSignalandNoise) #list of the files
    filesN = os.listdir(folderNoise)
    filesSN = [file for file in filesSN if os.path.isfile(os.path.join(folderSignalandNoise, file))]
    filesN = [file for file in filesN if os.path.isfile(os.path.join(folderNoise, file))]
    img_rows, img_cols, channels = imgDimensions  #define parameters for image size
    allNoise = np.zeros((img_rows, img_cols,len(filesN)))   # create numpy arrays to store all images from the folders
    allSignalNoise = np.zeros((img_rows, img_cols,len(filesSN))) 
    #loops to fill the allNoise and allSignalNoise arrays with the image data, this is done for testing, but also since it's easy to then convert these to png files for the pytorch/torchvision training
    count=0
    for file in filesN:
        pathNow = folderNoise + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allNoise[:,:,count] = Znow
        count += 1    
    count=0
    for file in filesSN:
        pathNow = folderSignalandNoise + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allSignalNoise[:,:,count] = Znow
        count += 1 
    #create an array to store the residual data => the signal    
    allResidues = allSignalNoise - allNoise   
    return allNoise, allSignalNoise, allResidues


def RAW_CSS_loader_NParrays(root_address, imgDimensions):
    folderSignalandNoise = DATM_GATM_address + 'raw/' ##signal+noise = RAW
    folderResidues = DATM_GATM_address + 'css/' ##residue = CSS stack result = approximation of interferomatric signal image
    filesSN = os.listdir(folderSignalandNoise) #list of the files
    filesR = os.listdir(folderResidues)
    filesSN = [file for file in filesSN if os.path.isfile(os.path.join(folderSignalandNoise, file))]
    filesN = [file for file in filesN if os.path.isfile(os.path.join(folderNoise, file))]
    filesR = [file for file in filesR if os.path.isfile(os.path.join(folderResidues, file))]
    img_rows, img_cols, channels = imgDimensions  #define parameters for image size
    allSignalNoise = np.zeros((img_rows, img_cols,len(filesSN))) 
    allResidues = np.zeros((img_rows, img_cols,len(filesSN)))
    #loops to fill the allNoise and allSignalNoise arrays with the image data, this is done for testing, but also since it's easy to then convert these to png files for the pytorch/torchvision training   
    count=0
    for file in filesSN:
        pathNow = folderSignalandNoise + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allSignalNoise[:,:,count] = Znow
        count += 1 
    count=0
    for file in filesR:
        pathNow = folderResidues + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allResidues[:,:,count] = Znow
        count += 1
    #create an array to store the residual data => the signal      
    allNoise = allSignalNoise - allResidues
    return allNoise, allSignalNoise, allResidues


# Function to normalize an array and convert it to uint8
def normalize_and_convert(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    return array.astype('uint8')


#loader to get datm/gatm address & prepare the directory for use in pytorch
def directoryPrep(DATM_GATM_address, imgDimensions, setting = 'DATM_GATM'):
    if setting == 'RAW_CSS':
        allNoise, allSignalNoise, allResidues = RAW_CSS_loader_NParrays(DATM_GATM_address, imgDimensions)
    elif setting == 'DATM_GATM':
        allNoise, allSignalNoise, allResidues = DATM_GATM_loader_NParrays(DATM_GATM_address, imgDimensions)
    #count the number of images of each type; noise, signal+noise, and residue
    #should be equal
    nCount = allNoise.shape[2]
    snCount = allSignalNoise.shape[2]
    rCount = allResidues.shape[2] 
    #organize process of saving images as png in the home directory, in a structure which works well with pytorch data loader
    folderBase = DATM_GATM_address  #'D:\\Research\\Yuri Fialko\\set 2/'
    nSub = 'NoisePng/'
    snSub = 'SignalNoisePng/'
    rSub = 'ResiduePng/'
    allFolder = folderBase + 'Png_n_sn_r/'
    nSubFull = folderBase + nSub
    snSubFull = folderBase + snSub
    rSubFull = folderBase + rSub
    #create directories if they do not exist yet
    if not os.path.exists(nSubFull):
        os.makedirs(nSubFull)
    if not os.path.exists(snSubFull):
        os.makedirs(snSubFull)
    if not os.path.exists(rSubFull):
        os.makedirs(rSubFull)
    if not os.path.exists(allFolder):
        os.makedirs(allFolder)
    # Save the images as PNG files
    for i in range(nCount):
        imgNow = allNoise[:,:,i]
        imgNow = normalize_and_convert(imgNow)
        imgNow2 = Image.fromarray(imgNow.astype('uint8'))
        imgNow2.save(nSubFull + 'n' + str(i) + '.png')
        imgNow2.save(allFolder + 'n' + str(i) + '.png')
    for i in range(snCount):
        imgNow = allSignalNoise[:,:,i]
        imgNow = normalize_and_convert(imgNow)
        imgNow2 = Image.fromarray(imgNow.astype('uint8'))
        imgNow2.save(snSubFull + 'sn' + str(i) + '.png')
        imgNow2.save(allFolder + 'sn' + str(i) + '.png')
    for i in range(rCount):
        imgNow = allResidues[:,:,i]
        imgNow = normalize_and_convert(imgNow)
        imgNow2 = Image.fromarray(imgNow.astype('uint8'))
        imgNow2.save(rSubFull + 'r' + str(i) + '.png')
        imgNow2.save(allFolder + 'r' + str(i) + '.png')
    print('Directories should be prepped with PNG files now')
    return None
    
    
#function to load in data from a folder with all the png files together, namely the folder saved previously "allFolder = folderBase + 'Png_n_sn_r/'"    
def organize_images_for_torchvision_APS(directory, train_test_split=0.8):
    # Create directories for the classes in train and test
    PNGallFolder =  directory+'Png_n_sn_r/'
    os.makedirs(os.path.join(PNGallFolder, 'train', 'class_0'), exist_ok=True)   #class_0 = residues
    os.makedirs(os.path.join(PNGallFolder, 'train', 'class_1'), exist_ok=True)   #class_1 = signal+noise
    os.makedirs(os.path.join(PNGallFolder, 'train', 'class_2'), exist_ok=True)   #class_2 = noise
    os.makedirs(os.path.join(PNGallFolder, 'test', 'class_0'), exist_ok=True)
    os.makedirs(os.path.join(PNGallFolder, 'test', 'class_1'), exist_ok=True)
    os.makedirs(os.path.join(PNGallFolder, 'test', 'class_2'), exist_ok=True)
    # Create a list of filenames for each class
    filenames_class_0 = [filename for filename in os.listdir(PNGallFolder) if filename.endswith('.png') and 'r' in filename]
    filenames_class_1 = [filename for filename in os.listdir(PNGallFolder) if filename.endswith('.png') and 'sn' in filename]
    filenames_class_2 = [filename for filename in os.listdir(PNGallFolder) if filename.endswith('.png') and 'r' not in filename and 'sn' not in filename]
    # Sort the filenames
    filenames_class_0.sort()
    filenames_class_1.sort()
    filenames_class_2.sort()
    # Calculate the train/test split index for each class
    split_index_class_0 = int(len(filenames_class_0) * train_test_split)
    split_index_class_1 = int(len(filenames_class_1) * train_test_split)
    split_index_class_2 = int(len(filenames_class_2) * train_test_split)
    # Split the filenames into train and test for each class
    train_filenames_class_0 = filenames_class_0[:split_index_class_0]
    test_filenames_class_0 = filenames_class_0[split_index_class_0:]
    train_filenames_class_1 = filenames_class_1[:split_index_class_1]
    test_filenames_class_1 = filenames_class_1[split_index_class_1:]
    train_filenames_class_2 = filenames_class_2[:split_index_class_2]
    test_filenames_class_2 = filenames_class_2[split_index_class_2:]
    # Move files to the appropriate class directories in the train folder
    for filename in train_filenames_class_0:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'train', 'class_0', filename))
    for filename in train_filenames_class_1:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'train', 'class_1', filename))
    for filename in train_filenames_class_2:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'train', 'class_2', filename))
    # Move files to the appropriate class directories in the test folder
    for filename in test_filenames_class_0:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'test', 'class_0', filename))
    for filename in test_filenames_class_1:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'test', 'class_1', filename))
    for filename in test_filenames_class_2:
        shutil.move(os.path.join(PNGallFolder, filename), os.path.join(PNGallFolder, 'test', 'class_2', filename))
    return None    


def loaderFULL(root_address, imgDimensions, train_test_ratio=0.8, setting='DATM_GATM'):
    if setting == 'DATM_GATM':
        directoryPrep(root_address, imgDimensions, setting)
        organize_images_for_torchvision_APS(root_address, train_test_split=train_test_ratio)
    elif setting == 'RAW_CSS':
        directoryPrep(root_address, imgDimensions, setting)
        organize_images_for_torchvision_APS(root_address, train_test_split=train_test_ratio)
    print('Data should be prepped into train/test subdirectories with class directories within each for noise, signal, and residue classes of images')
    return None
    

###########################################################################################################################################################################


###########################################################################################################################################################################
#The following code block encapsulates the machine learning class architecture based on the simpleViT example from https://github.com/lucidrains/vit-pytorch
#my model changes are within the 'SimpleViTaps' class where I have changed the model from classification to filtration/transform
#i left all the original architecture intact since why not & also might help if one wants to utilize the other ViT model examples & modify any of them please refer to differences between SimpleViT and SimpleViTaps

"""
Note, the following SimpleViT architecture has been sourced from https://github.com/lucidrains/vit-pytorch
My modifications are encapsulated within 'SimpleViTaps' where I change the classification model to transform/filtration via model head change
such that rather than returning classification vectors model returns image tensor for     loss( model( SignalAndNoise) , Residual = SignalAndNoise - Noise) [MSE loss]


this is done within:

self.linear_head = nn.Linear(self.dim, self.channels * self.image_height * self.image_width)

x = self.linear_head(x)
x = x.view(-1, self.channels, self.image_height, self.image_width)
return x  


The CustomDatasetAPS is my creation, but heavily inspired by examples in pytorch references
It serves as the system to pick files from the train/test => class_0/class_1/class_2/... folders and load them via the pytorch dataloader for batch training
One of the important functions of my code is the use of 'multiplicity' parameter within this dataset class
Please refer to class definition or documentation at bottom of page for details and advice regarding use of multiplicity parameter and CustomDatasetAPS
Basic idea is that it takes single image sets with SignalAndNoise and Residual, resamples spatial noise abd creates new channels containing these resamples, extending tensor
Meant to simulate method of CSS since can stack up images with resampled noise to have ML model pick up on the long lasting feature of signal
"""

# helpers (functions)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes for simpleViT architecture
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        x = self.transformer(x)
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        return self.linear_head(x)

    
class SimpleViTaps(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = self.channels * self.patch_height * self.patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.pos_embedding = posemb_sincos_2d(
            h = self.image_height // self.patch_height,
            w = self.image_width // self.patch_width,
            dim = self.dim,
        ) 
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        # Change this line to output a tensor with the same shape as the input
        self.linear_head = nn.Linear(self.dim, self.channels * self.image_height * self.image_width)

    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        x = self.transformer(x)
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        x = self.linear_head(x)
        # Reshape the output to match the input dimensions
        x = x.view(-1, self.channels, self.image_height, self.image_width)
        return x    
    
    
class CustomDatasetAPS(Dataset):
    def __init__(self, root_dir, transform=None, multiplicity=1):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample. In my code transformAPS is used, doesn't do much but left up to continued work of project to experiment there, check final notes section
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subs1 = {'train': 'train/', 'test': 'test/'}
        self.subs2 = {'residues': 'class_0/', 'SignalNoise': 'class_1/', 'Noise': 'class_2/'}
        self.FullPaths = {'TrainR': root_dir+self.subs1['train']+self.subs2['residues'],
                         'TestR': root_dir+self.subs1['test']+self.subs2['residues'],
                         'TrainSN': root_dir+self.subs1['train']+self.subs2['SignalNoise'],
                         'TestSN': root_dir+self.subs1['test']+self.subs2['SignalNoise'],
                         'TrainN': root_dir+self.subs1['train']+self.subs2['Noise'],
                         'TestN': root_dir+self.subs1['test']+self.subs2['Noise'],}
        self.TrainrFiles = os.listdir(self.FullPaths['TrainR'])
        self.TestrFiles = os.listdir(self.FullPaths['TestR'])
        self.TrainsnFiles = os.listdir(self.FullPaths['TrainSN'])
        self.TestsnFiles = os.listdir(self.FullPaths['TestSN'])
        self.TrainnFiles = os.listdir(self.FullPaths['TrainN'])
        self.TestnFiles = os.listdir(self.FullPaths['TestN'])
        self.TrainrFiles = [file for file in self.TrainrFiles if os.path.isfile(os.path.join(self.FullPaths['TrainR'], file))]
        self.TestrFiles = [file for file in self.TestrFiles if os.path.isfile(os.path.join(self.FullPaths['TestR'], file))]
        self.TrainsnFiles = [file for file in self.TrainsnFiles if os.path.isfile(os.path.join(self.FullPaths['TrainSN'], file))]
        self.TestsnFiles = [file for file in self.TestsnFiles if os.path.isfile(os.path.join(self.FullPaths['TestSN'], file))]
        self.TrainnFiles = [file for file in self.TrainnFiles if os.path.isfile(os.path.join(self.FullPaths['TrainN'], file))]
        self.TestnFiles = [file for file in self.TestnFiles if os.path.isfile(os.path.join(self.FullPaths['TestN'], file))]
        self.FullList = self.TrainrFiles + self.TestrFiles + self.TrainsnFiles + self.TestsnFiles + self.TrainnFiles + self.TestnFiles
        self.multiplicity = multiplicity
                
    def __len__(self):
        return len(self.TrainsnFiles)
     
    def _multiNoise_(self, SNimg):
        std = torch.std(SNimg)
        noisy_channels = []
        for _ in range(self.multiplicity):
            std = torch.std(SNimg)
            noise_samples = torch.randn(self.multiplicity, *SNimg.shape)
            noisy_channels = SNimg + noise_samples * std
            noisy_channels = noisy_channels.squeeze(1)
            return noisy_channels

    def _SignalStack_(self, Rimg):
        return Rimg.unsqueeze(1).expand(-1, self.multiplicity, -1, -1).squeeze(0)
    
    def __getitem__(self, idx):
        trainSNpath = self.FullPaths['TrainSN'] + self.TrainsnFiles[idx]
        trainRpath = self.FullPaths['TrainR'] + self.TrainrFiles[idx]
        SNimg = Image.open(trainSNpath)
        Rimg = Image.open(trainRpath)
        if self.transform:
            SNimg = self.transform(SNimg)
            Rimg = self.transform(Rimg)
        noisy_SNimg = self._multiNoise_(SNimg)
        SignalStack = self._SignalStack_(Rimg)
        return noisy_SNimg, SignalStack   
        
        
#what do i want to plot? (functions to call)
#plot example of original SNimg, original Rimg (both only 1 channel), Nimg = SNimg-Rimg next to each other. by using either the CustomDatasetAPS.__getitem__ or other standardized methods
#plot set of channels from noisy_SNimg, SignalStack from CustomDatasetAPS.__getitem__, plot them all in a large mosaic where user can specify how many images from noisy_SNimg, SignalStack tensors
#plot channel-wise average of noisy_SNimg, SignalStack from CustomDatasetAPS.__getitem__
#plot set of model(noisy_SNimg), SignalStack; in other words pass noisy_SNimg tensor through model
#plot (all batches) loss vs epoch   
        
###########################################################################################################################################################################        
#set of all plotting functions

#note that dataset item retrieval function __getitem__ retrieves SN image, R image, does the multiplicity based noise resample so thing being saved here is 
#(copy since long lasting)stack of Residuals and the resampled noise SNimg stack where channels=Multiplicity
#so save the thing across all new channels as a gif where each channel is a different image in gif sequence

#saves to root directory currently
def SaveGIF_inROOT_SN_R_N(datasetITEMidx, DEFAULT_DATA_ROOTaddress='/Users/evavra/Software/APS_removal_via_Pytorch_ViT/'): 
    transformAPS = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),])
    #set up dataset object and dataloader
    Classified_APS_Path = DEFAULT_DATA_ROOTaddress + 'Png_n_sn_r/'
    Classified_APS_Dataset = CustomDatasetAPS(root_dir=Classified_APS_Path, transform=transformAPS, multiplicity = Multiplicity)
    SNstack, Rstack = Classified_APS_Dataset.__getitem__(datasetITEMidx)
    Nstack = SNstack - Rstack
    SNstack = (SNstack - SNstack.min()) / (SNstack.max() - SNstack.min())
    Rstack = (Rstack - Rstack.min()) / (Rstack.max() - Rstack.min())
    Nstack = (Nstack - Nstack.min()) / (Nstack.max() - Nstack.min())
    frames = []
    for i in range(SNstack.shape[0]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('dataset item ' + str(datasetITEMidx) + ' channel 0 to ' + str(Multiplicity-1) + '[Multiplicity-1]')
        axs[0].imshow(SNstack[i], cmap='gray')
        axs[0].set_title('SNimg' + ' 0 to ' + str(SNstack.shape[0]-1))
        axs[1].imshow(Rstack[i], cmap='gray')
        axs[1].set_title('Rimg'+ ' 0 to ' + str(SNstack.shape[0]-1))
        axs[2].imshow(Nstack[i], cmap='gray')
        axs[2].set_title('Nimg'+ ' 0 to ' + str(SNstack.shape[0]-1))
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(DEFAULT_DATA_ROOTaddress+'imgTensor'+str(datasetITEMidx)+'.gif', frames, 'GIF', duration=0.1)
    return None

#function to plot average of resampled Signal+noise images, Noise images, and Residuals
def plot_channelAVGs(datasetITEMidx):
    SNstack, Rstack = Classified_APS_Dataset.__getitem__(datasetITEMidx)
    Nstack = SNstack - Rstack
    SNstack = (SNstack - SNstack.min()) / (SNstack.max() - SNstack.min())
    Rstack = (Rstack - Rstack.min()) / (Rstack.max() - Rstack.min())
    Nstack = (Nstack - Nstack.min()) / (Nstack.max() - Nstack.min())
    avg_SNstack = SNstack.mean(dim=0)
    avg_SignalStack = Rstack.mean(dim=0)
    avg_Nstack = Nstack.mean(dim=0)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(avg_SNstack, cmap='gray')
    axs[0].set_title('Average SNimg')
    axs[1].imshow(avg_SignalStack, cmap='gray')
    axs[1].set_title('Average SignalStack')
    axs[2].imshow(avg_Nstack, cmap='gray')
    axs[2].set_title('Average Nimg')
    plt.show()
    
#function to plot the signal averages across all the N=Multiplicity image channels, and compare with similar average across the model(Signal+Noise) result tensor
def plot_channelAVGsVSmodel(datasetITEMidx, epochNum, dataSet, Model, DEFAULT_DATA_ROOTaddress='/Users/evavra/Software/APS_removal_via_Pytorch_ViT/'):
    SNstack, Rstack = dataSet.__getitem__(datasetITEMidx)
    Nstack = SNstack - Rstack
    # Add an extra dimension for the batch size
    SNstack = SNstack.unsqueeze(0)
    ModelStack = Model(SNstack)
    # Remove the batch dimension for plotting
    ModelStack = ModelStack.squeeze(0)
    SNstack = SNstack.squeeze(0)
    SNstack = (SNstack - SNstack.min()) / (SNstack.max() - SNstack.min())
    Rstack = (Rstack - Rstack.min()) / (Rstack.max() - Rstack.min())
    Nstack = (Nstack - Nstack.min()) / (Nstack.max() - Nstack.min())
    ModelStack = (ModelStack - ModelStack.min()) / (ModelStack.max() - ModelStack.min())
    #print(SNstack.shape, Rstack.shape, Nstack.shape, ModelStack.shape)
    avg_SNstack = SNstack.mean(dim=0)
    avg_SignalStack = Rstack.mean(dim=0)
    avg_Nstack = Nstack.mean(dim=0)
    avg_ModelStack = ModelStack.mean(dim=0)
    #print(avg_SNstack.shape, avg_SignalStack.shape, avg_Nstack.shape, avg_ModelStack.shape)
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    fig.suptitle('dataset item ' + str(datasetITEMidx) + ' channel averages', fontsize=16, fontweight='bold', y=0.75, verticalalignment='center', horizontalalignment='center')
    axs[0].imshow(avg_SNstack, cmap='gray')
    axs[0].set_title('Average SNimg')
    axs[1].imshow(avg_SignalStack, cmap='gray')
    axs[1].set_title('Average SignalStack')
    axs[2].imshow(avg_Nstack, cmap='gray')
    axs[2].set_title('Average Nimg')
    #axs[3].imshow(avg_ModelStack, cmap='gray')
    axs[3].imshow(avg_ModelStack.detach().numpy(), cmap='gray')
    axs[3].set_title('Model Output AVG')
    plt.savefig(DEFAULT_DATA_ROOTaddress+'AVGcomparison'+str(datasetITEMidx)+'epoch'+str(epochNum)+'.png')
    plt.show()
    plt.clf()
    
    
#the following function is meant to encapsulate the entire process from datm/gatm directory management, data prep, training, and plotting
def trainModelFromScratch(rootDir, imgDim, train_test_ratio, setting, image_size, patch_size, dim, depth, heads, mlp_dim, Multiplicity, channels, dim_head,num_epochs, LearnRate, BatchNum, intermediateIDX=0, DEFAULT_DATA_ROOTaddress='/Users/evavra/Software/APS_removal_via_Pytorch_ViT/'):  
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    loaderFULL(rootDir, imgDim, train_test_ratio, setting)
    # Create an instance of the SimpleViT model
    model = SimpleViTaps(image_size=image_size,patch_size=patch_size,dim=dim,depth=depth,heads=heads,mlp_dim=mlp_dim,channels=channels,dim_head=dim_head)
    #transformation initially applied to each sample image as it is fetched by 
    transformAPS = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),])
    #set up dataset object and dataloader
    Classified_APS_Path = rootDir + 'Png_n_sn_r/'
    Classified_APS_Dataset = CustomDatasetAPS(root_dir=Classified_APS_Path, transform=transformAPS, multiplicity = Multiplicity)
    train_loader = DataLoader(Classified_APS_Dataset, batch_size=BatchNum, shuffle=True,drop_last=True) 
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Define the loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LearnRate)
    lossAVG = []
    for epoch in range(num_epochs):
        model.train()
        lossEpoch = []
        for SN, R in train_loader:
            SN, R = SN.to(device), R.to(device)
            optimizer.zero_grad()
            outputs = model(SN)
            loss = torch.sqrt(criterion(outputs, R))
            loss.backward()
            optimizer.step()
            # Append the loss to the list of losses
            lossEpoch.append(loss.item())
        lossAVG.append(np.mean(lossEpoch))
        plot_channelAVGsVSmodel(intermediateIDX, epochNum = epoch, dataSet = Classified_APS_Dataset, Model = model)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    modelFilename = 'APSmodel_' + timestamp
    torch.save(model, rootDir + modelFilename)
    # After training, plot the losses
    plt.plot(lossAVG)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('batch-avg Loss vs. Epoch during training')
    # Generate a unique filename with the current date and time
    filename = f'LossAVGvsEpoch_{timestamp}.png'
    # Save the plot with the updated filename
    plt.savefig(rootDir+filename)
    plt.show()
    plt.clf()
        
        
#useful if model already trained & want to conduct further training or use as initial condition for more testing or other purposes        
def loadTrainedModel(rootDir, modelFilename):
    # Later, when loading the model:
    loaded_model = torch.load(rootDir + modelFilename)
    
    
def load_data(DATM_GATM_address, imgDimensions):
    folderNoise = DATM_GATM_address + 'gatm/'
    folderSignalandNoise = DATM_GATM_address + 'datm/'
    filesSN = [file for file in os.listdir(folderSignalandNoise) if os.path.isfile(os.path.join(folderSignalandNoise, file))]
    filesN = [file for file in os.listdir(folderNoise) if os.path.isfile(os.path.join(folderNoise, file))]
    img_rows, img_cols, channels = imgDimensions
    allNoise = np.zeros((img_rows, img_cols,len(filesN)))
    allSignalNoise = np.zeros((img_rows, img_cols,len(filesSN)))

    count=0
    for file in filesN:
        pathNow = folderNoise + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allNoise[:,:,count] = Znow
        count += 1    

    count=0
    for file in filesSN:
        pathNow = folderSignalandNoise + file
        filNow = xr.open_dataset(pathNow, engine='netcdf4')
        Znow = filNow['z']
        allSignalNoise[:,:,count] = Znow
        count += 1 

    allResidues = allSignalNoise - allNoise
    return allNoise, allSignalNoise, allResidues


def plot_images(address, allNoise, allSignalNoise, allResidues,idx=0):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Sample: ' + str(idx))
    im1 = axs[0].imshow(allNoise[:,:,idx], cmap='gray')
    axs[0].set_title('Noise')
    fig.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(allSignalNoise[:,:,idx], cmap='gray')
    axs[1].set_title('Signal + Noise')
    fig.colorbar(im2, ax=axs[1])
    im3 = axs[2].imshow(allResidues[:,:,idx], cmap='gray')
    axs[2].set_title('Residues')
    fig.colorbar(im3, ax=axs[2])
    plt.savefig(address + 'sample' + str(idx) + '.png')
    plt.show()
    
    plt.close()
    

# usage
#DATM_GATM_address =DATM_GATM_address
#imgDimensions = [40,40,1]
#USERidx = 3
#allNoise, allSignalNoise, allResidues = load_data(DATM_GATM_address, imgDimensions)
#plot_images(DATM_GATM_address, allNoise, allSignalNoise, allResidues,idx = USERidx)


def plotMany(address, allNoise, allSignalNoise, allResidues, init, fin):
    for i in range (init,fin):
        plot_images(address, allNoise, allSignalNoise, allResidues,idx = i)
        
#plotMany(DATM_GATM_address, allNoise, allSignalNoise, allResidues,0,10)
     
if __name__ == '__main__':
    main()