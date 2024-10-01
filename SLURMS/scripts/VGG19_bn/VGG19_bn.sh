
export CUDA_VISIBLE_DEVICES=1


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn
echo "Created SLURMS/LOGS/VGG19_bn"


# echo "Started training ISIC2016_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016_VGG19_bn.err
# echo "Ended training ISIC2016_VGG19_bn
wait


# echo "Started training ISIC2017_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2017_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2017_VGG19_bn.err
# echo "Ended training ISIC2017_VGG19_bn
wait


# echo "Started training ISIC2018_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2018_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2018_VGG19_bn.err
# echo "Ended training ISIC2018_VGG19_bn
wait


# echo "Started training ISIC2019_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2019_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2019_VGG19_bn.err
# echo "Ended training ISIC2019_VGG19_bn
wait


# echo "Started training ISIC2020_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020_VGG19_bn.err
# echo "Ended training ISIC2020_VGG19_bn
wait


# echo "Started training _7_point_criteria_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria_VGG19_bn.err
# echo "Ended training _7_point_criteria_VGG19_bn
wait


# echo "Started training PAD_UFES_20_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PAD_UFES_20_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PAD_UFES_20_VGG19_bn.err
# echo "Ended training PAD_UFES_20_VGG19_bn
wait


# echo "Started training MEDNODE_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/MEDNODE_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/MEDNODE_VGG19_bn.err
# echo "Ended training MEDNODE_VGG19_bn
wait


# echo "Started training KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/KaggleMB_VGG19_bn.err
# echo "Ended training KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017_VGG19_bn
wait


# echo "Started training ISIC2018+ISIC2019_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2018+ISIC2019_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2018+ISIC2019_VGG19_bn.err
# echo "Ended training ISIC2018+ISIC2019_VGG19_bn
wait


# echo "Started training ISIC2020+PH2_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020+PH2_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020+PH2_VGG19_bn.err
# echo "Ended training ISIC2020+PH2_VGG19_bn
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20_VGG19_bn.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_VGG19_bn
wait


# echo "Started training MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_VGG19_bn
wait


# echo "Started training ISIC2019+ISIC2020+PH2_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2019+ISIC2020+PH2_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2019+ISIC2020+PH2_VGG19_bn.err
# echo "Ended training ISIC2019+ISIC2020+PH2_VGG19_bn
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG19_bn
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19_bn
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19_bn
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG19_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19_bn
wait

