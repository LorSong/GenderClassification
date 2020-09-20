import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

def plot_histories(history, epochs):
    """
    Takes history dictionary and plots Accuracy and Loss for training and validation sets.
    Arguments:
    history: dictionary with train/valid accuracy/loss
    epochs: number of epochs to draw
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axhline(max(val_acc), ls="--", c="g", label='Max accuracy = {}'.format(round(max(val_acc),3)))
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.title('Training and Validation Loss')
    plt.show()
    
def unbatchify(data, max_size=1000):
    """
    Takes a batched dataset of (image, label) tensors and returns
    two arrays of images and labels.
    Arguments:
    data: tf.dataset
    max_size: amount of images to return
    """
    images = []
    labels = []
    for i, (image, label) in enumerate(data.unbatch().as_numpy_iterator()):
        images.append(image.astype(int))
        labels.append(label.astype(int))
        if i + 2 > max_size:
            break 
    return np.array(images), np.array(labels).squeeze()

def plot_images(images, y_true=None, y_pred=None, class_names=[0,1], n_cols=5, figsize=(12,8)):
    """
    Takes images and plots them in a tiled manner.
    Labels can be provided for additional visualisation.
    Also accepts predicted probabilities for labels. Displays predicted class.
    
    Arguments:
    images: numpy arrays (shape: n_images, height, widths, color_channels)
    y_true: 1D array, optional - true target values
    y_pred: 1D array, optional - predicted probabilities (0-1)
    class_names: list, optional - these names will be shown as labels
    n_cols: number of images in one row
    figsize: matplotlib.pyplot figure parameter
    """
    n = len(images)
    n_rows = n // n_cols + 1
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(image)
        plt.axis('off')
        if (y_pred is not None) and (y_true is not None):
            # Change color if prediction right
            pred_class = class_names[np.round(y_pred[i]).astype(int)]
            true_class = class_names[np.round(y_true[i]).astype(int)]
            if pred_class == true_class:
                color = "green"
            else:
                color = "red"
            confidence = round(abs(0.5 - y_pred[i]) * 200, 2)
            plt.title("{}, {:2.0f}% confidence".format(pred_class,
                                                       confidence),
                                                color=color)

        elif y_true is not None:
            pred_class = class_names[np.round(y_true[i]).astype(int)]
            plt.title("{}".format(pred_class, color="yellow"))



def subfoldering(data_path):
    """
    Takes path to data and creates subfolders with 100 images each.
    """
    for gender in ["female", "male"]:
      folder_path = os.path.join(data_path, gender)

      with os.scandir(os.path.join(data_path, gender)) as files:
        for i,file in enumerate(files):
          if i % 100 == 0:
            new_dir = os.path.join(folder_path, "images_{}".format(str(i // 100 + 1).zfill(4)))
            print("new_dir", new_dir)
            if not os.path.isdir(new_dir):
              os.mkdir(new_dir)
          file_path = os.path.join(folder_path, file)
          if os.path.isfile(file_path):
            shutil.move(file_path, new_dir)

