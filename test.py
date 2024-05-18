import melanoma as mel

import itertools
import glob
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pathlib


CFG={
    'verbose': 1,
}

img_size = (384, 384)

snapshot_path = '/raid/mpsych/MELANOMA/snapshot/'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers'
# snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot/bestperformers/ensemble'

modelfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*_{img_size[0]}h_{img_size[1]}w_*.hdf5', recursive=True)]))
modelnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, modelfiles))
# modelpath = os.path.abspath(os.path.join(modelfiles, os.pardir))

networktypes = [net.name for net in mel.NetworkType]
# modelnames.reverse()



for idx, m in enumerate(modelnames):
    # netname = [each_m for each_m in networktypes if(each_m in m)]
    # assert len(netname) == 1
    # if len(netname) == 1:
    #     mapped_netname = commondata.DBpreprocessorDict[netname[0]]
    # elif len(netname) == 2:
    #     mapped_netname = commondata.DBpreprocessorDict[netname[1]]
    # if len(netname) == 1:
    #     netname = netname[0]
    # elif len(netname) == 2:
    #     netname = netname[1]
    # else:
    #     assert len(netname) <= 2
    dbpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001/melanomaDB/customDB/uniform01'

    dbpath_KaggleDB = os.path.join(dbpath, f'KaggleMB_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_HAM10000 = os.path.join(dbpath, f'HAM10000_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_ISIC2016 = os.path.join(dbpath, f'ISIC2016_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_ISIC2017 = os.path.join(dbpath, f'ISIC2017_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_ISIC2018 = os.path.join(dbpath, f'ISIC2018_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_ISIC2020 = os.path.join(dbpath, f'ISIC2020_{img_size[0]}h_{img_size[1]}w_binary.h5')
    dbpath_7pointcriteria = os.path.join(dbpath, f'_7_point_criteria_{img_size[0]}h_{img_size[1]}w_binary.h5')
    # final_perf.append(base_model.evaluate_model_onAll(model_name=m, model_path=snapshot_path, network_name = netname[0], dbpath_KaggleDB=dbpath_KaggleDB, dbpath_HAM10000=dbpath_HAM10000, \
    #     dbpath_ISIC2016=dbpath_ISIC2016, dbpath_ISIC2017=dbpath_ISIC2017, dbpath_ISIC2018=dbpath_ISIC2018, dbpath_7pointcriteria=dbpath_7pointcriteria))

    # mel.Model.evaluate_model_onAll(model_name=m, model_path=snapshot_path, dbpath_KaggleDB=dbpath_KaggleDB,\
    #                                 dbpath_HAM10000=dbpath_HAM10000, dbpath_ISIC2016=dbpath_ISIC2016, \
    #                                 dbpath_ISIC2017=dbpath_ISIC2017, dbpath_ISIC2018=dbpath_ISIC2018, \
    #                                 dbpath_7pointcriteria=dbpath_7pointcriteria)

mel.Model.extract_performances(snapshot_path)




print("finish")