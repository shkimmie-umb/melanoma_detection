import matplotlib.pyplot as plt
import pathlib
import glob

import pandas as pd

from tensorflow.keras.utils import plot_model
# from keras.utils.vis_utils import plot_model





class Visualizer:
    def __init__(self):
        pass

    def show_performance(history):
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            epochs_range = range(history.params["epochs"])
            
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='upper left')
            plt.title('Training and Validation Accuracy')
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper center')
            plt.title('Training and Validation Loss')
            plt.show()

    def data_viewer(path, class_names):
        ### visualize one instance of all the nine classes present in the dataset
        trainDataPath = pathlib.Path(path+'/Train')
        testDataPath = pathlib.Path(path+'/Test')
        # Plot train data samples
        plt.figure(figsize=(15,15))
        for i in range(len(class_names)):
            plt.subplot(3,3,i+1)
            image= plt.imread(str(list(trainDataPath.glob(class_names[i]+'/*.jpg'))[0]))
            plt.title(class_names[i])
            plt.imshow(image)
        # Todo: Plot test data samples

        # Show class distribution
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_axes([0,0,1,1])
        x=[]
        y=[]
        for i in range(len(class_names)):
            x.append(class_names[i])
            if (len(glob.glob(path+'/Train' + '/*/*/*.*'))) == 0:

                y.append(len(list(trainDataPath.glob(class_names[i]+'/*.jpg'))))
            else:
                y.append(len(list(trainDataPath.glob(class_names[i]+'/*.jpg'))) + len(list(trainDataPath.glob(class_names[i]+'/*/*.*'))))

        ax.bar(x,y)
        ax.set_ylabel('Numbers of images')
        ax.set_title('Class distribution of the different dermatology images (without augmentation) ')

        plt.xticks(rotation=45)
        plt.show()

        
        if (len(glob.glob(path+'/Train' + '/*/*/*.*'))) == 0:
            print("Number of samples for each class (without augmentation):")
            for i in range(len(class_names)):
                print(class_names[i],' - ',len(list(trainDataPath.glob(class_names[i]+'/*.jpg'))))
        else:
            print("Number of samples for each class (without augmentation):")
            for i in range(len(class_names)):
                print(class_names[i],' - ',len(list(trainDataPath.glob(class_names[i]+'/*.jpg'))))

            print("\nNumber of samples for each class (with augmentation):")
            for i in range(len(class_names)):
                print(class_names[i],' - ',len(list(trainDataPath.glob(class_names[i]+'/*.jpg')))+len(list(trainDataPath.glob(class_names[i]+'/*/*.*'))))

    def visualize_model(self, model, model_name):
        print(model.summary())
        return plot_model(model, to_file=f'{model_name}_plot.png', show_shapes=True, show_layer_names=True)


    def visualize_performance(self, model_name, history, fontsize = 14):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        metrics = history.history['accuracy']
        epochs_range = range(1, len(metrics) + 1) 

        plt.figure(figsize=(23, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.suptitle(f'Accuracy & Loss for {model_name} model', fontsize=fontsize)
        plt.show()

    