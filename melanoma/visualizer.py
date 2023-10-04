import matplotlib.pyplot as plt
import pathlib

class Visualizer:
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