# Melanoma-detection-CNN
> The aim of the project is to build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## Data Information
- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

- The data set contains the following diseases:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

Download link: https://drive.google.com/file/d/1v_Nfg3QD5_TIr3Y-awIBm7lTFmVtIvQj/view?usp=drive_link

## Conclusions
- The final model used the Augmentor library to artificially generate enough samples to come up with a model that can detect a correct cancer type with a validation accuracy around 84%. 
- The validation loss for the final model was 0.56

## Environment (chimera clean env)
- Keras - 2.5.0rc0
- Tensorflow - 2.5.0
- Augmentor - 0.2.10
- matplotlib==3.2.1
- pandas==1.2.0
- numpy==1.19.4
- pip install pydot
- conda install -c anaconda graphviz

## Contact
Created by [@shkimmie-umb] - feel free to contact me!
