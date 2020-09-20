# Imports
import importlib
import os
import sys
import json
# Silencing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Silencing tensorflow depreciation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

if tf.__version__ != "2.3.0":
    print("TF version is not 2.3.0, behavior may not be correct")

def find_images(folder_path):
    """
    Forms lists with paths to images and their filenames.
    """
    
    image_paths = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            
            image_paths.append(image_path)
            filenames.append(file)
    print("Found {} files".format(len(filenames)))       
    return image_paths, filenames

def process_image(image_path, img_size=(96,96)):
    """
     Takes an image path and returns a resized tensored image.
    """
    
    image = tf.io.read_file(image_path) # creates string tensor
    # turn jpeg image into numerical tensor with 3 color channels
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize image
    image = tf.image.resize(image, size=[img_size[0], img_size[1]])
    return image

def make_predictions(model, image_paths, filenames):
    """
    Takes a trained model and paths to images with their names.
    Returns predictions on them.
    """
    classes = ["female", "male"]
    # Form dataset from paths to images
    data = tf.data.Dataset.from_tensor_slices(image_paths)
    data = data.map(process_image).batch(16)
    
    print("Making predictions...")
    y_pred = model.predict(data).squeeze()
    # Transform predictions into {imagename: class} dictionary
    predicted_classes = [classes[tf.math.round(pred).numpy().astype(int)] 
                         for pred in y_pred]
    predicted_images = dict(zip(filenames,predicted_classes))
    return predicted_images 
   
    
def find_and_process_images(path_to_images):     
    #Looking for images
    image_paths, filenames = find_images(path_to_images)
    if image_paths:   
        print("Loading model...")
        try:
            mobilev2_model = tf.keras.models.load_model('./model')
        except:
            raise Exception("Failed to load the model. Must be in /model folder")
        
        # Making predictions and saving them
        predictions = make_predictions(mobilev2_model, image_paths, filenames)
        with open('process_results.json', 'w') as outfile:
            json.dump(predictions, outfile)
        print("Successfully created process_results.json")
    else:
        print("Error: no .jpg or .jpeg files were found. Check the path")

def main():
    # Taking path argument            
    try:
        path_to_images = sys.argv[1]
    except:
        path_to_images = "."
        print("Path to images is not provided, looking in the current folder")
    
    # Preventing memory errors with GPU (copied from TF documentation)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)
          
    find_and_process_images(path_to_images)
       
if __name__ == "__main__": 
    main()
    
    
    

        
