from .parser import *


class parser_7pointdb(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType._7_point_criteria.name

        self.makeFolders(datasetname)

        img_path = pathlib.Path(self.base_dir).joinpath('data', datasetname, 'release_v0', 'images')

        num_imgs = len(list(img_path.glob('*/*.[jpg][JPG]'))) # counts all 7-point db training images

        # assert num_imgs == 2013 # Num of files in folder

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)


        imagedir_dict = {os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x)): x for x in glob(os.path.join(img_path, '*/*.*'))}
        imagedir_dict_lower = {k.lower(): v for k, v in imagedir_dict.items()}
        # imagedir_dict_lower = list(map(lambda x: x.lower(), list(imagedir_dict.keys())))
        # imageid_path_dict_7pointdb = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*.*'))}

        # imagedir_dict_lower_dict = dict()
        # for ele in imagedir_dict_lower:
        # 	imagedir_dict_lower_dict[str(ele)] = ele
        
        df_7pointdb = pd.read_csv(str(pathlib.Path(img_path).joinpath('..')) + '/meta/meta.csv', header=0)

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
        df_7pointdb['cell_type_binary'] = df_7pointdb['diagnosis'].apply(lambda x: 'malignant' if 'melanoma' in x else 'benign')
        df_7pointdb['cell_type_binary_idx'] = pd.CategoricalIndex(df_7pointdb.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in 7 point db training metadata")
        display(df_7pointdb.isnull().sum())
        

        df_7pointdb['image'] = df_7pointdb.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_7pointdb.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)

        df_training_index = pd.read_csv(str(pathlib.Path(img_path).joinpath('..')) + '/meta/train_indexes.csv', header=0)
        df_validation_index = pd.read_csv(str(pathlib.Path(img_path).joinpath('..')) + '/meta/valid_indexes.csv', header=0)
        df_test_index = pd.read_csv(str(pathlib.Path(img_path).joinpath('..')) + '/meta/test_indexes.csv', header=0)
        # df_training_7pointdb = df_7pointdb[df_7pointdb.index.isin(df_training_index['indexes'])]
        df_training = df_7pointdb.filter(items = df_training_index['indexes'], axis=0)
        df_validation = df_7pointdb.filter(items = df_validation_index['indexes'], axis=0)
        df_test = df_7pointdb.filter(items = df_test_index['indexes'], axis=0)
        df_training.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['trainimages']
        df_validation.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['validationimages']
        df_test.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType._7_point_criteria]['testimages']
        

        # df_training_7pointdb['image'] = df_training_7pointdb.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))			

        mel.Preprocess().saveNumpyImagesToFiles(df_training, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_validation, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_test, self.test_rgb_folder)
        # 7 point db binary images/labels
        trainpixels = list(map(lambda x:x[0], df_training.image)) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], df_validation.image)) # Filter out only pixel from the list
        testpixels = list(map(lambda x:x[0], df_test.image)) # Filter out only pixel from the list
        
        trainids = list(map(lambda x:x[1].stem, df_training['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, df_validation['image']))
        testids = list(map(lambda x:x[1].stem, df_test['image']))

        trainlabels_binary = np.asarray(df_training.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(df_validation.cell_type_binary_idx, dtype='float64')
        testlabels_binary = np.asarray(df_test.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary_7pointdb = to_categorical(df_training_7pointdb.cell_type_binary_idx, num_classes=2)
        # validationlabels_binary_7pointdb = to_categorical(df_validation_7pointdb.cell_type_binary_idx, num_classes=2)
        # testlabels_binary_7pointdb = to_categorical(df_test_7pointdb.cell_type_binary_idx, num_classes=2)

        
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]
        # assert trainimages_7pointdb.shape[0] == trainlabels_binary_7pointdb.shape[0]
        # assert validationimages_7pointdb.shape[0] == validationlabels_binary_7pointdb.shape[0]
        # assert testimages_7pointdb.shape[0] == testlabels_binary_7pointdb.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

            
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