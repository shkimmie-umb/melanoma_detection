from .parser import *


class parser_KaggleMB(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.KaggleMB.name

        self.makeFolders(datasetname)

        dbpath = pathlib.Path(self.base_dir).joinpath('data', datasetname, 'Kaggle_malignant_benign_DB')

        num_imgs = len(list(dbpath.glob('t*/*/*.*'))) # counts all Kaggle Malignant Benign training images

        # train: 1440 benign, 1197 malignant; test: 360 benign + 300 malignant
        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages'] + mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        # imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(dbpath, 't*/*/*.*'))}
        paths = glob(os.path.join(dbpath, 't*/*/*.*'))
        # labels_dict = {os.path.basename(x): x for x in os.path.abspath(os.path.join(os.path.join(imageid_path_dict.values()), os.pardir))}
        df = pd.DataFrame()


        # Kaggle MB: Creating New Columns for better readability
        df['path'] = paths
        df['label'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir))))
        df['portion'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir, os.pardir))))
        assert df['label'].unique().shape[0] == 2
        # df['cell_type_binary'] = np.where(df['label'] == 'malignant', 'Melanoma', 'Non-Melanoma')
        df['cell_type_binary'] = df['label']
        df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in Kaggle MB training metadata")
        display(df.isnull().sum())
        
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

        # Dividing Kaggle MB into train/test set
        df_trainset_full = df.query('portion == "train"')
        df_testset = df.query('portion == "test"')

        # Dividing PAD UFES 20 into train/val set
        df_trainset, df_validationset = train_test_split(df_trainset_full, test_size=0.2,random_state = self.pseudo_num)
        

        mel.Preprocess().saveNumpyImagesToFiles(df_trainset, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_validationset, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_testset, self.test_rgb_folder)

        # KaggleMB binary images/labels
        trainpixels = list(map(lambda x:x[0], df_trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], df_validationset['image'])) # Filter out only pixel from the list
        testpixels = list(map(lambda x:x[0], df_testset['image'])) # Filter out only pixel from the list
        
        trainids = list(map(lambda x:x[1].stem, df_trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, df_validationset['image']))
        testids = list(map(lambda x:x[1].stem, df_testset['image']))
        
        
        trainlabels_binary = np.asarray(df_trainset.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(df_validationset.cell_type_binary_idx, dtype='float64')
        testlabels_binary = np.asarray(df_testset.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
        # validationlabels_binary = to_categorical(df_validationset.cell_type_binary_idx, num_classes=2)
        # testlabels_binary = to_categorical(df_testset.cell_type_binary_idx, num_classes=2)

        assert len(trainpixels)+len(validationpixels) == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages']
        assert len(testpixels) == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]
        # assert trainimages.shape[0] == trainlabels_binary.shape[0]
        # assert validationimages.shape[0] == validationlabels_binary.shape[0]
        # assert testimages.shape[0] == testlabels_binary.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)
            
    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']

        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on KaggleMB DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.KaggleMB.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.KaggleMB.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.KaggleMB.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.KaggleMB.name, \
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