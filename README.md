# COMP0055: Seeing is Not Believing
**Azizan Wazir, Nithin Anand, Fehed Wasti** \
*UCL MEng Computer Science*

Implementation of the strong and weak camouflage attack forms described by Xiao et. al. in their paper, [Seeing is Not Believing (2019)](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao).

This implementation generates attack images as described in the paper using DCCP and CVXPY in Python in Google Colab. Generated attack images are in the attack_images folder.

~~Currently, the attack images and the dimensions of the output image (dimensions to be resized to) are:~~

New attack images have been created and placed in the attack_images folder. All of these resize to (229, 229). The old attack images, output dimensions at the bottom of this README, are still available and are placed in a folder inside the attack_image folder.

## Updates
**16/03/2021:** We have found and proposed a new defence against attack images using random padding and cropping. This is demonstrated in the Colab notebook and has been found to work with all images we have tested on.

**22/02/2021:** Pushed a new version of the scaling_attack notebook as I had forgotten to add an extra parameter to the strong attack form's get perturbation function. Have tested it and it is fixed. Next push will contain weak attack form images.

## Colab notebook
**scaling_attack.ipynb**
This notebook consist of two parts. The first part consist of the theory involved, which is explained in depth in the paper. This consists of some pseudocode and mathematical explanations. This was mainly to assist us in the implementation, but is fun to know. The second part is the code used, separated into different parts, corresponding to steps in the execution. This is namely defining scaling functions, getting coefficients (conversion matrices), getting perturbations, the actual attack functions and main functions for executing the attack forms to generate attack images.


## **Running the program**
To run the notebook, open the notebook in Colab or Jupyter and either modify the code given and upload images to be used to a Google Drive folder of your choice, upload images locally to Colab or use local files in Jupyter. The last cell in the Setup cell defines paths for images in your chosen Google Drive folder. It's just for convenience. In the main functions, ```imread``` takes these paths and loads them. If you want to standardise the final size of the output image, you can resize ```tgt``` to your desired size before running the algorithm. The main functions will run their strong or weak attack form and display the attack image and output image. The commented line at the bottom is for saving the image locally (in Colab, in Jupyter it saves to the notebook local directory). In Colab, this image will not be saved to Drive or locally, and will need to be manually downloaded (from the left side directory explorer) to be saved.

Note: This is not a perfect implementation and was implemented for the purpose of a UCL coursework.



``` python
Attack Image    | Output dimensions

atk_jeff.jpg    | (250, 203)

atk_sheep.jpg   | (229, 229)

atk_ronnie.jpg  | (172, 229)

atk_cat.jpg     | (124, 229)
```
