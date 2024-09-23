from .parser import *


class parser_PH2(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.PH2.name

        self.makeFolders(datasetname)

        img_path = pathlib.Path(self.base_dir).joinpath('data', datasetname, 'PH2 Dataset images')

        num_imgs = len(list(img_path.glob('*/*_Dermoscopic_Image/*.bmp'))) # counts all PH2 training images

        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.PH2]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        imageid_path_dict_PH2 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*_Dermoscopic_Image/*.bmp'))}

        
        df_PH2 = pd.read_excel(str(pathlib.Path(img_path).joinpath('..', 'PH2_dataset.xlsx')), header=12)

        assert df_PH2.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.PH2]['trainimages']

        self.logger.debug("Let's check PH2 metadata briefly")
        self.logger.debug("This is PH2 data samples")
        display(df_PH2.head())

        # PH2: Creating New Columns for better readability
        df_PH2['path'] = df_PH2['Image Name'].map(imageid_path_dict_PH2.get)
        df_PH2['cell_type_binary'] = np.where(df_PH2['Melanoma'] == 'X', 'malignant', 'benign')
        df_PH2['cell_type_binary_idx'] = pd.CategoricalIndex(df_PH2.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in PH2 training metadata")
        display(df_PH2.isnull().sum())
        
        df_PH2['image'] = df_PH2.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_PH2.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)        

        mel.Preprocess().saveNumpyImagesToFiles(df_PH2, self.train_rgb_folder)

        # PH2 binary images/labels
        trainpixels = list(map(lambda x:x[0], df_PH2['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, df_PH2['image'])) # Filter out only pixel from the list
        
        trainlabels_binary = np.asarray(df_PH2.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary_PH2 = to_categorical(df_PH2.cell_type_binary_idx, num_classes=2)

        assert num_imgs == len(trainpixels)
        assert len(trainpixels) == trainlabels_binary.shape[0]
        # assert trainimages_PH2.shape[0] == trainlabels_binary_PH2.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)