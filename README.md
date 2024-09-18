# LNCR-DL-model
Deep Learning multiple staining renal biopsies image analysis to predict 12-month CR of induction treatment in LN patients 

### Slide digitalization and WSI pre-processing
Renal biopsy slides were scanned with KF-PRO-120-HI, KFBIO scanner at 40× magnification to generate digital WSIs (resolution 0.25μm/pixel) and saved as TIFF images. 
WSIs of each histological staining were first divided into nonoverlapping 512×512 pixels tiles under 10×, 20× and 40× magnifications using OpenSlide (3.4.1). 
Brightness and contrast of tiles were adjusted, and tiles containing more than 50% ( ＞ 225 pixels) non-tissue area were considered as invalid and excluded. 
Color normalization was performed for each histological staining using StainTools to minimize the staining variability.

### Training
Training script is in [Modeltrain.py] file. 
Models were trained with pretrained networks of AlexNet, DenseNet-121, Inception-V3, ResNet-50, VGG-11 and Vision Transformer or different batch sizes containing 8, 16, 32 and 64 tiles randomly sampled from a WSI. 
Each category was trained for a total of 10 times. For each time training, 100 epochs were performed. 
Then both the optimal CNN architecture for each histological staining and the most appropriate batch size were determined to train the final model for under different magnifications and different stainings.

Ensemble models of multi-magnification for signle-stain were obtained based on averaging the predicted value of each single-magnification model. Then, ensemble models of multi-stain were obtained based on maxed the predicted value of each ensemble single-stain model.

### Model Test
Model test script is in [ModelTest.py] file. Trained models were tested on the independent external test cohort.

### Tiles cluster & Model visualization prediction
Tile ckuster script is in [Cluster.py] file. 
To identify the key renal histopathological features that contribute the most to the model prediction, we performed unsupervised clustering of tiles using Deep Clustering with Category-Style representation (DCCS) and t-distributed stochastic neighbor embedding (t-SNE) algorithms. DCCS was employed to perform unsupervised clustering on the influential tiles, which it can aggregate the tiles into different clusters according to their features. Further, t-SNE was used for dimensionality reduction and cluster projection. 

Model visualization prediction script is in [gradCam.py] file. 
To identify the renal histopathological features related to induction treatment response, we applied gradient-weighted class activation mapping (gradCAM) to visualize the most predictive tiles.

