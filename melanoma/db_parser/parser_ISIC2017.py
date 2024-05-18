from .parser import *


class parser_ISIC2017(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        # ISIC2017
        self.lesion_type_dict_ISIC2017_task3_1 = { # Official ISIC2017 task 3 - 1
            0.0: 'nevus or seborrheic keratosis',
            1.0: 'melanoma'
        }
        self.lesion_type_dict_ISIC2017_task3_2 = { # Official ISIC2017 task 3 - 2
            0.0: 'melanoma or nevus',
            1.0: 'seborrheic keratosis',
        }
        self.lesion_type_binary_dict_ISIC2017 = { # Binary melanoma detection
            0.0: 'Non-Melanoma',
            1.0: 'Melanoma',
        }

        self.classes_ISIC2017_task3_1 = ['nevus or seborrheic keratosis', 'melanoma']
        self.classes_ISIC2017_task3_2 = ['melanoma or nevus', 'seborrheic keratosis']


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.ISIC2017.name

        self.makeFolders(datasetname)

        ISIC2017_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Training_Data')
        ISIC2017_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Validation_Data')
        ISIC2017_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Test_v2_Data')

        num_train_img_ISIC2017 = len(list(ISIC2017_training_path.glob('./*.jpg'))) # counts all ISIC2017 training images
        num_val_img_ISIC2017 = len(list(ISIC2017_val_path.glob('./*.jpg'))) # counts all ISIC2017 validation images
        num_test_img_ISIC2017 = len(list(ISIC2017_test_path.glob('./*.jpg'))) # counts all ISIC2017 test images

        assert num_train_img_ISIC2017 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages']
        assert num_val_img_ISIC2017 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages']
        assert num_test_img_ISIC2017 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']

        self.logger.debug('%s %s', "Images available in ISIC2017 train dataset:", num_train_img_ISIC2017)
        self.logger.debug('%s %s', "Images available in ISIC2017 validation dataset:", num_val_img_ISIC2017)
        self.logger.debug('%s %s', "Images available in ISIC2017 test dataset:", num_test_img_ISIC2017)

        # ISIC2017: Dictionary for Image Names
        imageid_path_training_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_training_path, '*.jpg'))}
        imageid_path_val_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_val_path, '*.jpg'))}
        imageid_path_test_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_test_path, '*.jpg'))}


        df_training_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Training_Part3_GroundTruth.csv')))
        df_val_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Validation_Part3_GroundTruth.csv')))
        df_test_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Test_v2_Part3_GroundTruth.csv')))


        self.logger.debug("Let's check ISIC2017 metadata briefly")
        self.logger.debug("This is ISIC2017 training data samples")
        # No need to create column titles (1st row) as ISIC2017 has default column titles
        display(df_training_ISIC2017.head())
        self.logger.debug("This is ISIC2017 test data samples")
        display(df_test_ISIC2017.head())

        

        # ISIC2017: Creating New Columns for better readability
        df_training_ISIC2017['path'] = df_training_ISIC2017.image_id.map(imageid_path_training_dict_ISIC2017.get)
        df_training_ISIC2017['cell_type_binary'] = df_training_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_training_ISIC2017['cell_type_task3_1'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_training_ISIC2017['cell_type_task3_2'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_training_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_training_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_training_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes

        df_val_ISIC2017['path'] = df_val_ISIC2017.image_id.map(imageid_path_val_dict_ISIC2017.get)
        df_val_ISIC2017['cell_type_binary'] = df_val_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_val_ISIC2017['cell_type_task3_1'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_val_ISIC2017['cell_type_task3_2'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_val_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_val_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_val_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes

        df_test_ISIC2017['path'] = df_test_ISIC2017.image_id.map(imageid_path_test_dict_ISIC2017.get)
        df_test_ISIC2017['cell_type_binary'] = df_test_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_test_ISIC2017['cell_type_task3_1'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_test_ISIC2017['cell_type_task3_2'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_test_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_test_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_test_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes



        self.logger.debug("Check null data in ISIC2017 training metadata")
        display(df_training_ISIC2017.isnull().sum())
        self.logger.debug("Check null data in ISIC2017 validation metadata")
        display(df_val_ISIC2017.isnull().sum())
        self.logger.debug("Check null data in ISIC2017 test metadata")
        display(df_test_ISIC2017.isnull().sum())

        df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(
        lambda x:(
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                         resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(
        lambda x:(
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                         resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(
        lambda x:(
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                         resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        assert all(df_training_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
        assert all(df_val_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
        labels = df_training_ISIC2017.cell_type_binary.unique()

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

        # ISIC2017 datasets are divided into train/val/test already
        trainset_ISIC2017 = df_training_ISIC2017
        validationset_ISIC2017 = df_val_ISIC2017
        testset_ISIC2017 = df_test_ISIC2017

        self.preprocessor.saveNumpyImagesToFiles(trainset_ISIC2017, df_training_ISIC2017, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_ISIC2017, df_val_ISIC2017, self.val_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(testset_ISIC2017, df_test_ISIC2017, self.test_rgb_folder)

        # ISIC2017 binary images/labels
        trainpixels_ISIC2017 = list(map(lambda x:x[0], trainset_ISIC2017['image'])) # Filter out only pixel from the list
        validationpixels_ISIC2017 = list(map(lambda x:x[0], validationset_ISIC2017['image'])) # Filter out only pixel from the list
        testpixels_ISIC2017 = list(map(lambda x:x[0], testset_ISIC2017['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset_ISIC2017['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, testset_ISIC2017['image']))
        validationids = list(map(lambda x:x[1].stem, validationset_ISIC2017['image']))


        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_ISIC2017 = to_categorical(trainset_ISIC2017.cell_type_binary_idx, num_classes=2)
        testlabels_binary_ISIC2017 = to_categorical(testset_ISIC2017.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_ISIC2017 = to_categorical(validationset_ISIC2017.cell_type_binary_idx, num_classes=2)

        assert num_train_img_ISIC2017 == len(trainpixels_ISIC2017)
        assert num_val_img_ISIC2017 == len(validationpixels_ISIC2017)
        assert num_test_img_ISIC2017 == len(testpixels_ISIC2017)
        assert len(trainpixels_ISIC2017) == trainlabels_binary_ISIC2017.shape[0]
        assert len(validationpixels_ISIC2017) == validationlabels_binary_ISIC2017.shape[0]
        assert len(testpixels_ISIC2017) == testlabels_binary_ISIC2017.shape[0]
        # assert trainimages_ISIC2017.shape[0] == trainlabels_binary_ISIC2017.shape[0]
        # assert validationimages_ISIC2017.shape[0] == validationlabels_binary_ISIC2017.shape[0]
        # assert testimages_ISIC2017.shape[0] == testlabels_binary_ISIC2017.shape[0]
        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_ISIC2017,
                        testpxs=testpixels_ISIC2017,
                        validationpxs=validationpixels_ISIC2017,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary_ISIC2017,
                        testlabels=testlabels_binary_ISIC2017,
                        validationlabels=validationlabels_binary_ISIC2017
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017],
            train_only=False,
            val_exists=True, 
            test_exists=True)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_ISIC2017_augmented, \
            trainlabels_binary_ISIC2017_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_ISIC2017,
                trainlabels=trainlabels_binary_ISIC2017,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_ISIC2017
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_ISIC2017_augmented, 
                            testpxs=testpixels_ISIC2017,
                            validationpxs=validationpixels_ISIC2017,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_binary_ISIC2017_augmented,
                            testlabels=testlabels_binary_ISIC2017,
                            validationlabels=validationlabels_binary_ISIC2017
                            )
            

    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on ISIC2017 DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.ISIC2017.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.ISIC2017.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2017.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2017.name, \
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