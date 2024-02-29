import melanoma as mel

import itertools
import glob
import os
import pathlib

CFG={
    'verbose': 1,
}

base_model = mel.CNN(CFG=CFG)

snapshot_path = '/raid/mpsych/MELANOMA/snapshot'

modelfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*.hdf5', recursive=True)]))
modelnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, modelfiles))
# modelpath = os.path.abspath(os.path.join(modelfiles, os.pardir))

img_size = (150, 150)
commondata = mel.CommonData()

networktypes = [net.name for net in mel.NetworkType]

final_perf = []
for idx, m in enumerate(modelnames):
    netname = [each_m for each_m in networktypes if(each_m in m)]
    assert len(netname) == 1
    mapped_netname = commondata.DBpreprocessorDict[netname[0]]
    dbpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001/melanomaDB/customDB/{mapped_netname}'

    dbpath_KaggleDB = dbpath+'/'+f'KaggleMB_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    dbpath_HAM10000 = dbpath+'/'+f'HAM10000_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    dbpath_ISIC2016 = dbpath+'/'+f'ISIC2016_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    dbpath_ISIC2017 = dbpath+'/'+f'ISIC2017_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    dbpath_ISIC2018 = dbpath+'/'+f'ISIC2018_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    dbpath_7pointcriteria = dbpath+'/'+f'_7_point_criteria_{img_size[0]}h_{img_size[1]}w_binary.pkl'
    final_perf.append(base_model.evaluate_model_onAll(model_name=m, model_path=snapshot_path, network_name = netname[0], dbpath_KaggleDB=dbpath_KaggleDB, dbpath_HAM10000=dbpath_HAM10000, \
        dbpath_ISIC2016=dbpath_ISIC2016, dbpath_ISIC2017=dbpath_ISIC2017, dbpath_ISIC2018=dbpath_ISIC2018, dbpath_7pointcriteria=dbpath_7pointcriteria))

