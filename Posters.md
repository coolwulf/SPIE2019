## AI / Deep Learning / Machine Learning
### Identifying disease-free chest x-ray images with deep transfer learning 
- datasets: 
    - NIH chestX0ray14
    - MIMIC-CXR
- X. Wang CVPR 2017
- P.  stanford
- Normal /abnormal classification
    - Minimie false positives
- Framework
    - Transfer learning
    - Inception-ResNet-v2
    - Dilated ResNet Block
    - Tesla P100 (16G)
- Remove 50% normal cases
  ![](spie/95.png)
  ![](spie/96.png)

### Analysis of deep convolutional features for detection of lung nodules in computed tomography 
- Feature extraction
- InceptionV4 / Inception-ResNet
- Feature embedding
- UMAP of radiomic features
- RRHO Maps / True positives

### Low-dose CT count-domain denoising via convolutional neural network with filter loss 
- filter-loss 
- count-domain
- Simple U-Net
- filter-loss / more relevant to the reconstruction
- MAE loss

### Artifact-driven sampling schemes for robust female pelvis CBCT segmentation using deep learning
- Varian
- Adaptive Radiotherapy (ART)
- Delivery of the dose over several factions
- Change in bladder volume
- Automation required
- CBCT artefact
- lower dose / scatter / Respiratory motion (longer scan times)
- Leading to Noise / lower contrast / streaking artefacts
- Air enclosed in bowel
- Goal: design a sampling scheme 
- class imbalance / selcective sampling of misclassifed samples / sampling of difficult image regions
- Bladder / Rectum / Uterus
- Data: CBCT: 21 patients, 351 scans, retro contoured
- CT: 92 patients.
- 2D U-Net: 
    - 4  resolution levels / Receptive field: 92x92
    - 4 output channels
- Training on 188^2 patches of axial slices resampled to 2.5mm^3 isotropic resolution
- Idea of Curriculum Learning
    - start with easy task )training on artifact free patches
    - increase difficulty of task (sample patches with artifacts)
    - Integration of domain knowledge
- decrease learning rate
- Indirect artiface estimation
    - artifacts are mainly caused by respiratory motion during acquisition with air within the bowel
    - compute body mask -> threshold at 300 HU -> exclude air in rectum -> compute volume per slice
    - Use volume of air / slice
- Estimated air distribution
- GDS descrease 1-4, better with curriculum
- Slice-wise dice scores
- Progress during curriculum training
    - Overall best DNN is saved within 30k iterations
    - trade-off between organs
    - no further imporvement when traiong only on slices with air / artifacts
- Performance varies strongly on slices w/ vs. wo/ artifacts
- Integration of domain knowledge may be helpful
- curriculum learning more promising than fixed sampling ratio ofr improving on regions w/ artifacts while maintaining performance on those wo/
- limitations hand-crafted scheme, noise
- visibility of organ contrours on slices affacted by severe artifacts
- future work: direct estimation and sampling of artifacts, 3D training

### Combining deep learning methods and human knowledge to identify abnormalities in computed tomography (CT) reports 
- Using text-based CT reports -> Developing a text-classification model -> Active learning
- Goal: develop a model identifies abnormalities within the CR reports and requires a significant reduced set of leabeled report
- clinical history / fidndings / impression / signature
- Inclusion criteria : ct of the chest abdomen and pelvis / Findings and impressions sections
- Data cleaning: eleminate test befrore findings / signature
- create binary leabels for every organ
- Develope a classification model -> Training the model -> select report w/ highest uncertainty reduction -> clinician create label for the selected report -> Add new dat a to rtraining set
- Label embedding Attentive Model (LEAM)
    - Report X: sequence of words
    - label y 1 abnormal , 0=normal
    - f0 word embedding: maps words to vectors
    - calculate attention score b)
    - f1 average of word embeddings weighted by attention score: Z = f0(x)*B
    - f2 multilayer perceptron predictiong y
- selection criteria : 
    - criteria: minimize the uncertainty of the model
    - measured uncertain through the covariance of the parameters
- Active learning requires less iterations than random sampling to achieve a coomparable performance
    - random sampling
    - simulated active learning
    - active learning

### Ensemble 3D residual network (E3D-ResNet) for reduction of false-positive polyp detections in CT colonography 
- E3D-ResNet  AUC value at 0.984

### Two-level training of a 3D U-Net for accurate segmentation of the intra-cochlear anatomy in head CT with limited ground truth training data 
- Condition GAN to add restraint / robutness of the model



## Posters
![](spie/68.png)
![](spie/69.png)
![](spie/70.png)
![](spie/71.png)
![](spie/72.png)
![](spie/73.png)
![](spie/74.png)
![](spie/75.png)
![](spie/76.png)
![](spie/77.png)
![](spie/78.png)
![](spie/79.png)
![](spie/97.png)
![](spie/98.png)
![](spie/121.png)
![](spie/122.png)
![](spie/123.png)
![](spie/124.png)
![](spie/125.png)
![](spie/126.png)
![](spie/127.png)
![](spie/128.png)
![](spie/129.png)