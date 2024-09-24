from .parser import *


class parser_PAD_UFES_20(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.PAD_UFES_20.name

        self.makeFolders(datasetname)

        db_path = pathlib.Path(self.base_dir).joinpath(datasetname)
        img_path = pathlib.Path(db_path).joinpath('images')

        num_imgs = len(list(img_path.glob('imgs_part_*/*.*'))) # counts all PAD_UFES_20 training images

        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(img_path, 'imgs_part_*/*.*'))}

        
        df = pd.read_csv(str(db_path) + '/metadata.csv', header=0)

        assert df.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20]['trainimages']

        self.logger.debug("Let's check PAD UFES 20 metadata briefly")
        self.logger.debug("This is PAD UFES 20 data samples")
        display(df.head())



        # PAD UFES 20: Creating New Columns for better readability
        df['path'] = df['img_id'].map(imageid_path_dict.get)
        df['cell_type_binary'] = np.where(df['diagnostic'] == 'MEL', 'malignant', 'benign')
        df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in PAD UFES 20 training metadata")
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
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)


        # df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

        # Dividing PAD UFES 20 into train/val set
        trainset, validationset = train_test_split(df, test_size=0.2,random_state = self.pseudo_num)
        

        mel.Preprocess().saveNumpyImagesToFiles(trainset, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(validationset, self.val_rgb_folder)

        # PAD UFES 20 binary images/labels
        trainpixels = list(map(lambda x:x[0], trainset.image)) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], validationset.image)) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset['image']))


        
        trainlabels_binary = np.asarray(trainset.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(validationset.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary_PAD_UFES_20 = to_categorical(trainset.cell_type_binary_idx, num_classes=2)
        # validationlabels_binary_PAD_UFES_20 = to_categorical(validationset.cell_type_binary_idx, num_classes=2)

        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20]['trainimages']
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        # assert trainimages_PAD_UFES_20.shape[0] == trainlabels_binary_PAD_UFES_20.shape[0]
        # assert validationimages_PAD_UFES_20.shape[0] == validationlabels_binary_PAD_UFES_20.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)