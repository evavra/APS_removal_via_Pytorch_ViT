# APS_removal_via_Pytorch_ViT
Pytorch ViT model to filter APS noise from InSAR images, utilizing CSS methodology via 'multiplicity' param


Title:VIT_for_APSremoval
Author: Luke Fairbanks

Credit: Yuri Fialko for project guidance & data; https://sioviz.ucsd.edu/~fialko/     
---the original training data for this ML model was from some matlab code of his meant to replicate APS (atmospherica phase screen) noise in InSAR images     
---the testing/development dataset is relatively simple, 40x40 images with two classes (datm - signal and noise, gatm - noise)   
---where the signal is a simple gaussian bump & the noise is randomized spatial noise across the pixels of the image, obscurring the underlying signal   
---one of key points is that this noise screen varies with time whereas the underlying signal is expected to be a more persistent signal wrt time  
---hence the structure of the ML model to take advantage of 'multiplicity' parameter to simulate 'moving window' of data    

Credit: Pytorch libraries & others which form basis for model; https://pytorch.org/     

Credit: https://github.com/lucidrains/vit-pytorch authors & the parent libraries they pull from;      
--they organized a host of ViT architectures as examples, these example files should be in the 'vit_pytorch' folder, more info at bottom     
--my ViT is different in some fundamental ways, namely it does not do classification but rather filtration/transform    
---the whole point is to take InSAR images, or in this case test images, and identify the persistent signal (fault line movement of Earth's crust)   
---where the noise over the image 

--effectively what my model does is take as input an image which has signal and noise, passes the image through a visual transfromer, the compares the output, the transformed image, to the residual signal which is SignalAndNoise - Noise data
--comparison is done via MSE across the pixel values, where model(SignalAndNoise) and Residues have same dimensionality
--then this MSE loss is used as the model training loss

--code is structured to create proper dimensions but for clarity tensors have dimensionality (batchNum, channelNum, IMGheight, IMGwidth)
--I'm going to try & automate as much as I can, and will show where user input is expected to best of my ability
--goal is to minimize it down to data folder root address input and parameter inputs

--'multiplicity' is an important parameter to use methodology of CSS (APS removal) since it simulates taking 1 resuidue/signal image(long lasting in time), and re-sampling spatial noise across the corresponding SignalAndNoise image
--the whole idea being one may increase 'multiplicity' (integer), increasing channelNum of tensor, allowing model to train/test on arbitrary length (temporal-simulated/noise-resampled) sequences of images
--this procedure effectively allows ViT model to look at 'moving window' of data as if from previous N=multiplicity image snapshots to extract the long lasting features of the image
--pixel variance of original SignalAndNoise image is used to re-sample noise w/ pytorch methods,

--check last section for more details on 'what next?'
--after you've got the code running on example data provided, please continue running cells
--contact Luke Fairbanks for questions; otherwise try code debuggers or chatbot-assisted debug/testing, pytorch/torchvision documentation, example code from git/lucidrains & other similar projects, etc...

If you wish to run the code on simply the example data and get a nice 'clean run' of the code, please [within the jupyter notebook] load imports below, run the function/class definition block, then proceed to [section 4] to train the model via the function 'trainModelFromScratch'

--Cheers
