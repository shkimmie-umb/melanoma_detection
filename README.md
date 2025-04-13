# Melanoma Detection with Uncertainty Quantification
[IEEE ISBI 2025 (Oral Presentation)](https://arxiv.org/pdf/2411.10322)
<p></p>
<img src="https://github.com/shkimmie-umb/melanoma_detection/blob/master/Thumbnail.png" width="480">
<!-- ![screenshot](https://github.com/shkimmie-umb/melanoma_detection/blob/master/Thumbnail.jpeg) -->

## Interactive Web Application
[Try the melanoma detector](https://mpsych.github.io/melanoma/)

## Supported datasets
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [ISIC2016](https://challenge.isic-archive.com/data/#2016)
- [ISIC2017](https://challenge.isic-archive.com/data/#2017)
- [ISIC2018](https://challenge.isic-archive.com/data/#2018)
- [ISIC2019](https://challenge.isic-archive.com/data/#2019)
- [ISIC2020](https://challenge.isic-archive.com/data/#2020)
- [PH2](https://www.fc.up.pt/addi/ph2%20database.html)
- [7-point criteria dataset](https://derm.cs.sfu.ca/Welcome.html)
- [PAD_UFES_20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
- [MED-NODE](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/)
- [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

## [SIIM-ISIC leaderboard results](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview)
| DBs | Network | Img size | Private Score <sup id="privatescore">[1](#privatescore)</sup> | Public Score <sup id="publicscore">[2](#publicscore)</sup> |
| ------------- | ------------- | ------------- | ------------- | ------------- |
|  ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+</br>PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+Kaggle | EfficientNetB1  | 224x224 | 0.9115  | 0.9063  |
|  ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+</br>PAD_UFES_20+MEDNODE | EfficientNetB1  | 224x224 | 0.9069  | 0.9068  |
|  ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+</br>PH2+_7_point_criteria+PAD_UFES_20+MEDNODE | DenseNet201  | 224x224 | 0.9061  | 0.9020  |
|  ISIC2016+ISIC2020 | EfficientNetB2  | 224x224 | 0.9046  | 0.9145  |
|  ISIC2016+ISIC2018+ISIC2019+ISIC2020 | ResNet152  | 224x24 | 0.9032  | 0.8975  |
|  ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+</br>PH2 | ResNet152  | 224x224 | 0.9007  | 0.9040  |
| ISIC2016+ISIC2018+ISIC2019+ISIC2020 | EfficientNetB1 | 224x224 | 0.9004 | 0.9057 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+</br>PH2+_7_point_criteria+MEDNODE+KaggleMB | ResNet101 | 224x224 | 0.8996 | 0.8868 |
<!-- |  Multiple <sup id="a1">[3](#dataset)</sup> | Ensemble <sup id="a2">[4](#network)</sup>  | 150x150 | 0.7618  | 0.7621  | -->

#### Note
> <sup id="privatescore">1</sup> Score on all test data. The potential winner(s) are determined solely by the leaderboard ranking on the private leaderboard. <br>
> <sup id="publicscore">2</sup> Score on the public testsets for reference (30% of partial test data) <br>
<!-- > <sup id="dataset">3</sup> Averaged the models in the table, trained with multiple datasets <br>
> <sup id="network">4</sup> Averaged the probabilities from the models in the table<br> -->

<!-- - ISIC contains the following diseases:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

- Original Download link: https://challenge.isic-archive.com/data/
- Folder-structured custom db: https://drive.google.com/file/d/1v_Nfg3QD5_TIr3Y-awIBm7lTFmVtIvQj/view?usp=drive_link -->

## Environment
- PyTorch 2.4.0
<!-- - Keras - 2.5.0rc0 -->
<!-- - Tensorflow - 2.5.0 -->
<!-- - Augmentor - 0.2.10 -->
- matplotlib==3.9.1
- pandas==2.2.2
- numpy==1.26.4

## Contact
sanghyuk.kim001@umb.edu
