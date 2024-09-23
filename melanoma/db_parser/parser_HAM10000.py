from .parser import *


class parser_HAM10000(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
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
            'bkl'  : 'benign',
            'nv'   : 'benign', # nevus
            'df'   : 'benign',
            'mel'  : 'malignant',
            'vasc' : 'benign',
            'bcc'  : 'benign',
            'akiec': 'benign',
        }


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.HAM10000.name

        self.makeFolders(datasetname)

        db_path = pathlib.Path(self.base_dir).joinpath('data', datasetname)
        num_train_img = len(list(db_path.glob('HAM10000_images_part_*/*.jpg'))) # counts all HAM10000 images

        self.logger.debug('%s %s', "Images available in HAM10000 train dataset:", num_train_img)

        # HAM10000: Dictionary for Image Names
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(db_path, 'HAM10000_images_part_*/*.jpg'))}

        df = pd.read_csv(str(pathlib.Path(db_path).joinpath('HAM10000_metadata.csv')))

        self.logger.debug("Let's check HAM10000 metadata briefly -> df.head()")
        # logger.debug("Let's check metadata briefly -> df.head()".format(df.head()))
        # print("Let's check metadata briefly -> df.head()")
        display(df.head())

        classes_multi_HAM10000 = df.dx.unique() # dx column has labels
        num_classes_multi_HAM10000 = len(classes_multi_HAM10000)
        # self.CFG_num_classes = num_classes
        classes_multi_HAM10000, num_classes_multi_HAM10000

        # Not required for pickled data
        # HAM10000: Creating New Columns for better readability
        df['num_images'] = df.groupby('lesion_id')["image_id"].transform("count")
        df['path'] = df.image_id.map(imageid_path_dict.get)
        df['cell_type'] = df.dx.map(self.lesion_type_dict_HAM10000.get)
        df['cell_type_binary'] = df.dx.map(self.lesion_type_binary_dict_HAM10000.get)

        # Define codes for compatibility among datasets
        # df['cell_type_idx'] = pd.Categorical(df.dx).codes
        df['cell_type_idx'] = pd.CategoricalIndex(df.dx, categories=['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']).codes
        # df['cell_type_binary_idx'] = pd.Categorical(df.cell_type_binary).codes
        df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes
        self.logger.debug("Let's add some more columns on top of the original metadata for better readability")
        self.logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx'")
        self.logger.debug("Now, let's show some of records -> df.sample(5)")
        display(df.sample(10))

        # Check null data in metadata
        self.logger.debug("Check null data in HAM10000 metadata -> df.isnull().sum()")
        display(df.isnull().sum())



        # We found there are some null data in age category
        # Filling in with average data
        # self.logger.debug("HAM10000: We found there are some null data in age category. Let's fill them with average data\n")
        # self.logger.debug("df.age.fillna((df.age.mean()), inplace=True) --------------------")
        # df.age.fillna((df.age.mean()), inplace=True)


        # # Now, we do not have null data
        # self.logger.debug("HAM10000: Let's check null data now -> print(df.isnull().sum())\n")
        # self.logger.debug("HAM10000: There are no null data as below:")
        # display(df.isnull().sum())
        
        df['image'] = df.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)

        # Dividing HAM10000 into train/val/test set
        
        df_single_HAM10000 = df[df.num_images == 1]
        trainset1_HAM10000, testset_HAM10000 = train_test_split(df_single_HAM10000, test_size=self.split_ratio, random_state = self.pseudo_num)
        trainset2_HAM10000, validationset_HAM10000 = train_test_split(trainset1_HAM10000, test_size=self.split_ratio, random_state = 4)
        trainset3_HAM10000 = df[df.num_images != 1]
        trainset_HAM10000 = pd.concat([trainset2_HAM10000, trainset3_HAM10000])

        mel.Preprocess().saveNumpyImagesToFiles(trainset_HAM10000, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(validationset_HAM10000, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(testset_HAM10000, self.test_rgb_folder)

        trainpixels = list(map(lambda x:x[0], trainset_HAM10000['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], validationset_HAM10000['image']))
        testpixels = list(map(lambda x:x[0], testset_HAM10000['image']))

        trainids = list(map(lambda x:x[1].stem, trainset_HAM10000['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset_HAM10000['image']))
        testids = list(map(lambda x:x[1].stem, testset_HAM10000['image']))
        
        trainlabels_multi = np.asarray(trainset_HAM10000.cell_type_idx, dtype='float64')
        validationlabels_multi = np.asarray(validationset_HAM10000.cell_type_idx, dtype='float64')
        testlabels_multi = np.asarray(testset_HAM10000.cell_type_idx, dtype='float64')
        trainlabels_binary = np.asarray(trainset_HAM10000.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(validationset_HAM10000.cell_type_binary_idx, dtype='float64')
        testlabels_binary = np.asarray(testset_HAM10000.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary_HAM10000 = to_categorical(trainset_HAM10000.cell_type_binary_idx, num_classes= 2)
        # testlabels_binary_HAM10000 = to_categorical(testset_HAM10000.cell_type_binary_idx, num_classes= 2)
        # validationlabels_binary_HAM10000 = to_categorical(validationset_HAM10000.cell_type_binary_idx, num_classes= 2)

        assert num_train_img == (len(trainpixels) + len(testpixels) + len(validationpixels))
        assert len(trainpixels) == trainlabels_multi.shape[0]
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_multi.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_multi.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]
        # assert trainimages_HAM10000.shape[0] == trainlabels_binary_HAM10000.shape[0]
        # assert validationimages_HAM10000.shape[0] == validationlabels_binary_HAM10000.shape[0]
        # assert testimages_HAM10000.shape[0] == testlabels_binary_HAM10000.shape[0]				
        

        # Unpack all image pixels using asterisk(*) with dimension (shape[0])
        # trainimages_HAM10000 = trainimages_HAM10000.reshape(trainimages_HAM10000.shape[0], *image_shape)
        
            
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