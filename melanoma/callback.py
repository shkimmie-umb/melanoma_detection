import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 10
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1
        
    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch % self.SHOW_NUMBER == 0) or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + "{:.4f}".format(logs['loss']) + ' accuracy: ' + "{:.4f}".format(logs['accuracy'])
                + ' val_loss: ' + "{:.4f}".format(logs['val_loss']) + ' val_accuracy: ' + "{:.4f}".format(logs['val_accuracy']))