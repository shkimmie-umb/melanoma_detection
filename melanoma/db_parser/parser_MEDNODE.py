from .parser import *


class parser_MEDNODE(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.MEDNODE.name

        self.makeFolders(datasetname)

        dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './complete_mednode_dataset')

        num_imgs = len(list(dbpath.glob('*/*.*'))) # counts all Kaggle Malignant Benign training images

        # train: 70 melanoma, 100 naevus
        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        # imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(dbpath, 't*/*/*.*'))}
        paths = glob(os.path.join(dbpath, '*/*.*'))
        # labels_dict = {os.path.basename(x): x for x in os.path.abspath(os.path.join(os.path.join(imageid_path_dict.values()), os.pardir))}
        df = pd.DataFrame()


        # MEDNODE: Creating New Columns for better readability
        df['path'] = paths
        df['label'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir))))
        # df['portion'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir, os.pardir))))
        # assert df['label'].unique().shape[0] == 2
        df['cell_type_binary'] = np.where(df['label'] == 'melanoma', 'Melanoma', 'Non-Melanoma')
        df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in Kaggle MB training metadata")
        display(df.isnull().sum())
        
        df['image'] = df.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)
                
        if not self.isWholeFeatureExist or not self.isTrainFeatureExist or not self.isValFeatureExist or not self.isTestFeatureExist:
            for i in labels:
                os.makedirs(f"{self.whole_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_feature_folder}/{i}", exist_ok=True)

        
        # Dividing MEDNODE into train/val set
        df_trainset, df_validationset = train_test_split(df, test_size=0.2,random_state = self.pseudo_num)
        

        self.preprocessor.saveNumpyImagesToFiles(df_trainset, df, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(df_validationset, df, self.val_rgb_folder)
        # preprocessor.saveNumpyImagesToFiles(df_testset, df, test_rgb_folder)

        # MEDNODE binary images/labels
        trainpixels = list(map(lambda x:x[0], df_trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], df_validationset['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, df_trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, df_validationset['image']))        
        
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
        validationlabels_binary = to_categorical(df_validationset.cell_type_binary_idx, num_classes=2)
        # testlabels_binary = to_categorical(df_testset.cell_type_binary_idx, num_classes=2)

        assert len(trainpixels)+len(validationpixels) == 70+100
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        # assert trainimages.shape[0] == trainlabels_binary.shape[0]
        # assert validationimages.shape[0] == validationlabels_binary.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels,
                        testpxs=[],
                        validationpxs=validationpixels,
                        trainids=trainids, 
                        testids=[],
                        validationids=validationids,
                        trainlabels=trainlabels_binary,
                        testlabels=[],
                        validationlabels=validationlabels_binary
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE],
            train_only=False,
            val_exists=True, 
            test_exists=False)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_PADUFES20_augmented, \
            trainlabels_binary_PADUFES20_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels,
                trainlabels=trainlabels_binary,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_trainset
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_PADUFES20_augmented, 
                            testpxs=[],
                            validationpxs=validationpixels,
                            trainids=trainids_new, 
                            testids=[],
                            validationids=validationids,
                            trainlabels=trainlabels_binary_PADUFES20_augmented,
                            testlabels=[],
                            validationlabels=validationlabels_binary
                            )
            
    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.MEDNODE]['testimages']

        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on MEDNODE DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.MEDNODE.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.MEDNODE.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.MEDNODE.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.MEDNODE.name, \
                target_network = target_network, \
                    testlabels = np.array(testdata['testlabels']), test_pred_classes = test_pred_classes
        )

        performance = {
            'y_pred': test_pred_classes.tolist(),
            'accuracy': test_report['accuracy'],
            'precision': test_report['macro avg']['precision'],
            'sensitivity': test_report['Malignant']['recall'],
            'specificity': test_report['Benign']['recall'],
            'f1-score': test_report['macro avg']['f1-score'],
        }

        return performance