# Web-based Melanoma Detection
[ArXiv link](https://arxiv.org/abs/2403.14898)
<p></p>
<img src="https://github.com/shkimmie-umb/melanoma_detection/blob/master/Thumbnail.jpeg" width="480">
<!-- ![screenshot](https://github.com/shkimmie-umb/melanoma_detection/blob/master/Thumbnail.jpeg) -->


## Standalone Melanoma web application
[Melanoma web application](https://mpsych.github.io/melanoma/)

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
|  ISIC16'+ISIC17'+ISIC18'+ISIC19'+MEDNODE+Kaggle | DenseNet121  | 150x150 | 0.7211  | 0.7472  |
|  ISIC18' | ResNet50  | 150x150 | 0.5999  | 0.6301  |
|  ISIC20' | ResNet50  | 150x150 | 0.7751  | 0.8126  |
|  ISIC16'+ISIC17'+ISIC18'+ISIC19'+ISIC20'+PH2 | ResNet152  | 150x150 | 0.8064  | 0.8073  |
|  ISIC19' | ResNet152  | 150x150 | 0.6769  | 0.7234  |
|  ISIC16'+ISIC17'+ISIC18'+ISIC19'+ISIC20'+PH2+ <br> 7pointcriteria+PAD_UFES_20+MEDNODE+Kaggle | ResNet152  | 150x150 | 0.7774  | 0.7894  |
|  Multiple <sup id="a1">[3](#dataset)</sup> | Ensemble <sup id="a2">[4](#network)</sup>  | 150x150 | 0.7618  | 0.7621  |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2 | ResNet152 | 384x384 | 0.7028 | 0.7134 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019 | DenseNet169 | 384x384 | 0.6943 | 0.7471 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB | DenseNet169 | 384x384 | 0.7963 | 0.8535 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB | DenseNet169 | 384x384 | 0.8028 | 0.8247 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE | DenseNet169 | 384x384 | 0.7980 | 0.8338 |
| ISIC2016+ISIC2017+ISIC2018+_7_point_criteria+PAD_UFES_20 | ResNet50 | 384x384 | 0.4199 | 0.4484 |
| IC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2 | ResNet152 | 384x384 | 0.7028 | 0.7234 |
| ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE | ResNet152V2 | 384x384 | 0.8144 | 0.8284 |
| ISIC2016+MEDNODE | Xception | 384x384 | 0.7409 | 0.7474 |

#### Note
> <sup id="privatescore">1</sup> Score on 70% of private testsets. The potential winner(s) are determined solely by the leaderboard ranking on the private leaderboard. <br>
> <sup id="publicscore">2</sup> Score on the public testsets for reference <br>
> <sup id="dataset">3</sup> Averaged the models in the table, trained with multiple datasets <br>
> <sup id="network">4</sup> Averaged the probabilities from the models in the table<br>

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
sanghyuk.kim001@umb.edu
