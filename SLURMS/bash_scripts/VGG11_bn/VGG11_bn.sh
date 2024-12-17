
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn
echo "Created SLURMS/LOGS/VGG11_bn"


# echo "Started training ISIC2018_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2018_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2018_VGG11_bn.err
# echo "Ended training ISIC2018_VGG11_bn
wait


# echo "Started training ISIC2019_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2019_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2019_VGG11_bn.err
# echo "Ended training ISIC2019_VGG11_bn
wait


# echo "Started training ISIC2020_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2020_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2020_VGG11_bn.err
# echo "Ended training ISIC2020_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2020_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2020_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2020_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2020_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2020+PH2_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2020+PH2_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2020+PH2_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2020+PH2_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG11_bn
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11_bn > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11_bn/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11_bn
wait

