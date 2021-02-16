'''
    Image Crafting: Strong Attack Form

    Input:
        scaling function    Scale()
        source image        src_img
        target image        tgt_img
        source image size   src_width, src_height
        target image size   tgt_width, tgt_height

    Output:
        attack image        atk_img

    Pseudocode:

        m  = src_height
        n  = src_width
        m' = tgt_height
        n' = tgt_width

        CL, CR = GetCoefficient(m, n, m', n')

        // pertubation matrix of vertical attack
        delta_v = zeros(m,n') 
        
        // intermediate source image
        src_img* = Scale(src_img)

        for col = 0 to n' - 1:
            // vertical scaling attack
            delta_v[:,col] = GetPerturbation(src_img*[:,col], T[:,col], CL, obj='min')

        atk_img* = unsigned int(src_img* + delta_v)

        // pertubation matrix of horizontal attack
        delta_h = zeros(m,n)

        for row = 0 to m - 1:
            // horizontal scaling attack
            delta_h[row,:] = GetPerturbation(src_img[row,:], atk_img*[row,:], CR, obj='min')
        
        atk_img = unsigned int(src_img + delta_h)

    GetCoefficient:
        Recovering coefficients (5.3)
        
        It was established that some common scaling algorithms can be represented by the following eq:
            Scale(X) = CLm'm * Xmn * CRnn'
        where CL scales horizontally (m -> m') and CR scales vertically (n -> n') to give an output of (m',n')

        CLm'm * (Imm * INmax) = CLm'm * INMax
        (Inn * INMax) * CRnn' = CRnn' * INMax
        INMax = maximum pixel value for image format

        By using the identity matrix, CL and CR can be found (above). 

        This is done by setting the source image as src_img = Imm * INMax and scaling it to an (m',m) size matrix 
        (which negates the need for CR to be applied) to obtain D = Scale(src_img) = unsigned int(CLm'm * INMax) -> CLm'm ~= D/INMax
 
        Approximation of CL is CLm'm normalised for each row:
            For each row in CL, divide each element by the sum of all elements in the row (sum of row = 1)
    
        Repeat for CR.

    GetPerturbation:
        The perturbation functions (delta_1 and delta_2) are the differences between atk_img and src_img and the final output image (out_img) and the tgt_img
        This is described by the following equations
        
        atk_img = src_img + delta_1
        out_img = tgt_img + delta_2, where out_img = Scale(atk_img)

        The L-infinity norm of delta_2, the largest element magnitude in delta_2 (i.e. the largest difference in pixel intensity between out_img and tgt_img for any given pixel)
            has a threshold, given by e * INMax, where e (epsilon) is the specified contraint and INmax is the maximum pixel value for the image format (e.g. 255)
            
        || delta_2 ||inf <= e * INmax 

        The objective function for the strong attack form is to minimise the square of the L2 norm of delta_1. 
        The L2 norm of a matrix is equal to the square root of the sum of squares of each element, i.e.
            || x || = sqrt(sum([element^2 for element in x])), where || x || is the L2 norm of x
        
        We are minimising the sum of squares of each element in the matrix, which represent the difference between the atk_img and src_img
        So we are minimising the difference between the images, such that || delta_2 ||inf stays within the b oundaries of 0 and e * INMax

        min(|| delta_1 ||^2 )

        This can be represented as follows:
        Objective: min(|| delta_1 ||^2)

'''
import matplotlib as plt

import torch
from torchvision import datasets, transforms

import helper

img = 'dickbutt.jpg'
image_folder = 'D:\\Documents\\COMP0055\\comp0055_seeing_is_not_believing\\img\\'

img = image_folder + img

dataset = datasets.ImageFolder(image_folder, transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images,labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)