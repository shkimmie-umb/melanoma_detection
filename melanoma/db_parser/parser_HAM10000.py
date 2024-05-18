from .parser import *


class parser_HAM10000(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        # HAM10000
        self.lesion_type_dict_HAM10000 = {
            'bkl'  : 'Pigmented Benign keratosis',
            'nv'   : 'Melanocytic nevi', # nevus
            'df'   : 'Dermatofibroma',
            'mel'  : 'Melanoma',
            'vasc' : 'Vascular lesions',
            'bcc'  : 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
        }

        self.lesion_type_binary_dict_HAM10000 = {
            'bkl'  : 'Non-Melanoma',
            'nv'   : 'Non-Melanoma', # nevus
            'df'   : 'Non-Melanoma',
            'mel'  : 'Melanoma',
            'vasc' : 'Non-Melanoma',
            'bcc'  : 'Non-Melanoma',
            'akiec': 'Non-Melanoma',
        }


    def saveDatasetToFile(self, augment_ratio=None):
        
        datasetname = mel.DatasetType.HAM10000.name

        self.makeFolders(datasetname)

        HAM10000_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './HAM10000_images_combined')
        num_train_img_HAM10000 = len(list(HAM10000_path.glob('./*.jpg'))) # counts all HAM10000 images

        self.logger.debug('%s %s', "Images available in HAM10000 train dataset:", num_train_img_HAM10000)

        # HAM10000: Dictionary for Image Names
        imageid_path_dict_HAM10000 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(HAM10000_path, '*.jpg'))}

        df_HAM10000 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './HAM10000_metadata.csv')))

        self.logger.debug("Let's check HAM10000 metadata briefly -> df.head()")
        # logger.debug("Let's check metadata briefly -> df.head()".format(df.head()))
        # print("Let's check metadata briefly -> df.head()")
        display(df_HAM10000.head())

        classes_multi_HAM10000 = df_HAM10000.dx.unique() # dx column has labels
        num_classes_multi_HAM10000 = len(classes_multi_HAM10000)
        # self.CFG_num_classes = num_classes
        classes_multi_HAM10000, num_classes_multi_HAM10000

        # Not required for pickled data
        # HAM10000: Creating New Columns for better readability
        df_HAM10000['num_images'] = df_HAM10000.groupby('lesion_id')["image_id"].transform("count")
        df_HAM10000['path'] = df_HAM10000.image_id.map(imageid_path_dict_HAM10000.get)
        df_HAM10000['cell_type'] = df_HAM10000.dx.map(self.lesion_type_dict_HAM10000.get)
        df_HAM10000['cell_type_binary'] = df_HAM10000.dx.map(self.lesion_type_binary_dict_HAM10000.get)

        # Define codes for compatibility among datasets
        # df_HAM10000['cell_type_idx'] = pd.Categorical(df_HAM10000.dx).codes
        df_HAM10000['cell_type_idx'] = pd.CategoricalIndex(df_HAM10000.dx, categories=['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']).codes
        # df_HAM10000['cell_type_binary_idx'] = pd.Categorical(df_HAM10000.cell_type_binary).codes
        df_HAM10000['cell_type_binary_idx'] = pd.CategoricalIndex(df_HAM10000.cell_type_binary, categories=self.classes_melanoma_binary).codes
        self.logger.debug("Let's add some more columns on top of the original metadata for better readability")
        self.logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx'")
        self.logger.debug("Now, let's show some of records -> df.sample(5)")
        display(df_HAM10000.sample(10))

        # Check null data in metadata
        self.logger.debug("Check null data in HAM10000 metadata -> df_HAM10000.isnull().sum()")
        display(df_HAM10000.isnull().sum())



        # We found there are some null data in age category
        # Filling in with average data
        self.logger.debug("HAM10000: We found there are some null data in age category. Let's fill them with average data\n")
        self.logger.debug("df.age.fillna((df_HAM10000.age.mean()), inplace=True) --------------------")
        df_HAM10000.age.fillna((df_HAM10000.age.mean()), inplace=True)


        # Now, we do not have null data
        self.logger.debug("HAM10000: Let's check null data now -> print(df.isnull().sum())\n")
        self.logger.debug("HAM10000: There are no null data as below:")
        display(df_HAM10000.isnull().sum())
        
        df_HAM10000['image'] = df_HAM10000.path.map(
            lambda x:(
            # img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
            # img := load_img(path=x, target_size=(img_width, img_height)), # [0]: Encoded PIL object
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                         resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_HAM10000.cell_type_binary.unique()

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
            
        

        # Dividing HAM10000 into train/val/test set
        
        df_single_HAM10000 = df_HAM10000[df_HAM10000.num_images == 1]
        trainset1_HAM10000, testset_HAM10000 = train_test_split(df_single_HAM10000, test_size=self.split_ratio, random_state = self.pseudo_num)
        trainset2_HAM10000, validationset_HAM10000 = train_test_split(trainset1_HAM10000, test_size=self.split_ratio, random_state = 4)
        trainset3_HAM10000 = df_HAM10000[df_HAM10000.num_images != 1]
        trainset_HAM10000 = pd.concat([trainset2_HAM10000, trainset3_HAM10000])

        self.preprocessor.saveNumpyImagesToFiles(trainset_HAM10000, df_HAM10000, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_HAM10000, df_HAM10000, self.val_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(testset_HAM10000, df_HAM10000, self.test_rgb_folder)

        trainpixels_HAM10000 = list(map(lambda x:x[0], trainset_HAM10000['image'])) # Filter out only pixel from the list
        testpixels_HAM10000 = list(map(lambda x:x[0], testset_HAM10000['image']))
        validationpixels_HAM10000 = list(map(lambda x:x[0], validationset_HAM10000['image']))

        trainids = list(map(lambda x:x[1].stem, trainset_HAM10000['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, testset_HAM10000['image']))
        validationids = list(map(lambda x:x[1].stem, validationset_HAM10000['image']))

        # trainimages_HAM10000 = self.preprocessor.normalizeImgs(imgs=trainpixels_HAM10000, networktype=networktype,
        #                                             uniform_normalization=uniform_normalization)
        # validationimages_HAM10000 = preprocessor.normalizeImgs(imgs=validationpixels_HAM10000, networktype=networktype,
        #                                                 uniform_normalization=uniform_normalization)
        # testimages_HAM10000 = preprocessor.normalizeImgs(imgs=testpixels_HAM10000, networktype=networktype,
        #                                         uniform_normalization=uniform_normalization)

        
        trainlabels_multi_HAM10000 = np.asarray(trainset_HAM10000.cell_type_idx, dtype='float64')
        testlabels_multi_HAM10000 = np.asarray(testset_HAM10000.cell_type_idx, dtype='float64')
        validationlabels_multi_HAM10000 = np.asarray(validationset_HAM10000.cell_type_idx, dtype='float64')
        # trainlabels_binary_HAM10000 = np.asarray(trainset_HAM10000.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_HAM10000 = np.asarray(testset_HAM10000.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_HAM10000 = np.asarray(validationset_HAM10000.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_HAM10000 = to_categorical(trainset_HAM10000.cell_type_binary_idx, num_classes= 2)
        testlabels_binary_HAM10000 = to_categorical(testset_HAM10000.cell_type_binary_idx, num_classes= 2)
        validationlabels_binary_HAM10000 = to_categorical(validationset_HAM10000.cell_type_binary_idx, num_classes= 2)

        assert num_train_img_HAM10000 == (len(trainpixels_HAM10000) + len(testpixels_HAM10000) + len(validationpixels_HAM10000))
        assert len(trainpixels_HAM10000) == trainlabels_multi_HAM10000.shape[0]
        assert len(trainpixels_HAM10000) == trainlabels_binary_HAM10000.shape[0]
        assert len(validationpixels_HAM10000) == validationlabels_multi_HAM10000.shape[0]
        assert len(validationpixels_HAM10000) == validationlabels_binary_HAM10000.shape[0]
        assert len(testpixels_HAM10000) == testlabels_multi_HAM10000.shape[0]
        assert len(testpixels_HAM10000) == testlabels_binary_HAM10000.shape[0]
        # assert trainimages_HAM10000.shape[0] == trainlabels_binary_HAM10000.shape[0]
        # assert validationimages_HAM10000.shape[0] == validationlabels_binary_HAM10000.shape[0]
        # assert testimages_HAM10000.shape[0] == testlabels_binary_HAM10000.shape[0]				


        # Feature saving
        # for idx, order in enumerate(testset_HAM10000.index):
        # 	img = array_to_img(testimages_HAM10000[idx])
        # 	label = testset_HAM10000.cell_type_binary[order]
        # 	assert label == df_HAM10000.cell_type_binary[order]
        # 	img.save(f"{test_feature_folder}/{label}/{testset_HAM10000.image[order][2].stem}.jpg", quality=100, subsampling=0)

        # Convert into bytes
        # trainimages_bytes = self.PILtoBytes(trainimages_HAM10000)
        # testimages_bytes = self.PILtoBytes(testimages_HAM10000)
        # validationimages_bytes = self.PILtoBytes(validationimages_HAM10000)
        

        # Unpack all image pixels using asterisk(*) with dimension (shape[0])
        # trainimages_HAM10000 = trainimages_HAM10000.reshape(trainimages_HAM10000.shape[0], *image_shape)
        
        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width

        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_HAM10000,
                        testpxs=testpixels_HAM10000,
                        validationpxs=validationpixels_HAM10000,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary_HAM10000,
                        testlabels=testlabels_binary_HAM10000,
                        validationlabels=validationlabels_binary_HAM10000
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename, 
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000], 
            train_only=True,
            val_exists=False, 
            test_exists=False)



        # filename_multi = f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_multiclass.pkl' # height x width



        # Augmentation only on training set
        

        if augment_ratio is not None and augment_ratio >= 1.0:
            
            df_mel_augmented, df_non_mel_augmented, trainpixels_augmented, \
            trainlabels_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_HAM10000, 
                trainlabels=trainlabels_binary_HAM10000,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_HAM10000
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_augmented, 
                            testpxs=testpixels_HAM10000, 
                            validationpxs=validationpixels_HAM10000,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_augmented,
                            testlabels=testlabels_binary_HAM10000,
                            validationlabels=validationlabels_binary_HAM10000
                            )
            
    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.HAM10000]['testimages']

        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on HAM10000 DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.HAM10000.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.HAM10000.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.HAM10000.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.HAM10000.name, \
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

        return performance, model