from .parser import *


class parser_7pointdb(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType._7_point_criteria.name

        self.makeFolders(datasetname)

        _7pointdb_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './release_v0')

        img_path =pathlib.Path.joinpath(_7pointdb_path, './images')

        num_imgs = len(list(img_path.glob('*/*.*'))) # counts all 7-point db training images

        assert num_imgs == 2013 # Num of files in folder

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)


        imagedir_dict = {os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x)): x for x in glob(os.path.join(img_path, '*/*.*'))}
        imagedir_dict_lower = {k.lower(): v for k, v in imagedir_dict.items()}
        # imagedir_dict_lower = list(map(lambda x: x.lower(), list(imagedir_dict.keys())))
        # imageid_path_dict_7pointdb = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*.*'))}

        # imagedir_dict_lower_dict = dict()
        # for ele in imagedir_dict_lower:
        # 	imagedir_dict_lower_dict[str(ele)] = ele
        
        df_7pointdb = pd.read_csv(str(_7pointdb_path) + '/meta/meta.csv', header=0)

        assert df_7pointdb.shape[0] == 1011 # meta rows

        self.logger.debug("Let's check 7-point criteria db metadata briefly")
        self.logger.debug("This is 7-point criteria db samples")
        display(df_7pointdb.head())

        # 7 point criteria db: Creating New Columns for better readability
        # df_7pointdb['path_clinic'] = df_7pointdb['clinic'].str.lower().map(imagedir_dict_lower.get)
        df_7pointdb['path'] = df_7pointdb['derm'].str.lower().map(imagedir_dict_lower.get)
        # df_7pointdb['path_clinic'].shape[0] == 1011
        df_7pointdb['path'].shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages']\
            + mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages']\
                + mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']
        df_7pointdb['cell_type_binary'] = df_7pointdb['diagnosis'].apply(lambda x: 'Melanoma' if 'melanoma' in x else 'Non-Melanoma')
        df_7pointdb['cell_type_binary_idx'] = pd.CategoricalIndex(df_7pointdb.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in 7 point db training metadata")
        display(df_7pointdb.isnull().sum())
        

        df_7pointdb['image'] = df_7pointdb.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_7pointdb.cell_type_binary.unique()

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
                os.makedirs(f"{self.val_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_feature_folder}/{i}", exist_ok=True)

        df_training_index = pd.read_csv(str(_7pointdb_path) + '/meta/train_indexes.csv', header=0)
        df_validation_index = pd.read_csv(str(_7pointdb_path) + '/meta/valid_indexes.csv', header=0)
        df_test_index = pd.read_csv(str(_7pointdb_path) + '/meta/test_indexes.csv', header=0)
        # df_training_7pointdb = df_7pointdb[df_7pointdb.index.isin(df_training_index['indexes'])]
        df_training_7pointdb = df_7pointdb.filter(items = df_training_index['indexes'], axis=0)
        df_validation_7pointdb = df_7pointdb.filter(items = df_validation_index['indexes'], axis=0)
        df_test_7pointdb = df_7pointdb.filter(items = df_test_index['indexes'], axis=0)
        df_training_7pointdb.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages']
        df_validation_7pointdb.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages']
        df_test_7pointdb.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']
        

        # df_training_7pointdb['image'] = df_training_7pointdb.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))			

        # preprocessor.saveNumpyImagesToFiles(df_training_7pointdb, df_7pointdb, train_rgb_folder)
        # preprocessor.saveNumpyImagesToFiles(df_validation_7pointdb, df_7pointdb, val_rgb_folder)
        # preprocessor.saveNumpyImagesToFiles(df_test_7pointdb, df_7pointdb, test_rgb_folder)

        # 7 point db binary images/labels
        trainpixels_7pointdb = list(map(lambda x:x[0], df_training_7pointdb.image)) # Filter out only pixel from the list
        validationpixels_7pointdb = list(map(lambda x:x[0], df_validation_7pointdb.image)) # Filter out only pixel from the list
        testpixels_7pointdb = list(map(lambda x:x[0], df_test_7pointdb.image)) # Filter out only pixel from the list
        
        trainids = list(map(lambda x:x[1].stem, df_training_7pointdb['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, df_test_7pointdb['image']))
        validationids = list(map(lambda x:x[1].stem, df_validation_7pointdb['image']))
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_7pointdb = to_categorical(df_training_7pointdb.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_7pointdb = to_categorical(df_validation_7pointdb.cell_type_binary_idx, num_classes=2)
        testlabels_binary_7pointdb = to_categorical(df_test_7pointdb.cell_type_binary_idx, num_classes=2)

        
        assert len(trainpixels_7pointdb) == trainlabels_binary_7pointdb.shape[0]
        assert len(validationpixels_7pointdb) == validationlabels_binary_7pointdb.shape[0]
        assert len(testpixels_7pointdb) == testlabels_binary_7pointdb.shape[0]
        # assert trainimages_7pointdb.shape[0] == trainlabels_binary_7pointdb.shape[0]
        # assert validationimages_7pointdb.shape[0] == validationlabels_binary_7pointdb.shape[0]
        # assert testimages_7pointdb.shape[0] == testlabels_binary_7pointdb.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_7pointdb,
                        testpxs=testpixels_7pointdb,
                        validationpxs=validationpixels_7pointdb,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary_7pointdb,
                        testlabels=testlabels_binary_7pointdb,
                        validationlabels=validationlabels_binary_7pointdb
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria],
            train_only=False,
            val_exists=True, 
            test_exists=True)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_7pointdb_augmented, \
            trainlabels_binary_7pointdb_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_7pointdb,
                trainlabels=trainlabels_binary_7pointdb,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_7pointdb
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_7pointdb_augmented, 
                            testpxs=testpixels_7pointdb,
                            validationpxs=validationpixels_7pointdb,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_binary_7pointdb_augmented,
                            testlabels=testlabels_binary_7pointdb,
                            validationlabels=validationlabels_binary_7pointdb
                            )
            
    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']

        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        # 7 point criteria Testing
        print('Testing on 7-point-criteria DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType._7_point_criteria.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType._7_point_criteria.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType._7_point_criteria.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType._7_point_criteria.name, \
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