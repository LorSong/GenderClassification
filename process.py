def find_images(folder_path):
    """
    Forms lists with paths to images and their filenames
    """
    
    image_paths = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            
            image_paths.append(image_path)
            filenames.append(file)
            
    return image_paths, filenames

def process_image(image_path, img_size=(96,96)):
    """
     Takes image path and returns reshaped tensor
    """
    
    image = tf.io.read_file(image_path) # creates string tensor
    # turn jpeg image into numerical tensor with 3 color channels
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize image
    image = tf.image.resize(image, size=[img_size[0], img_size[1]])
    return image

def make_predictions(model, image_paths, filenames):
    """
    Takes pathes to images and trained model.
    Returns predictions on them
    """
    classes = ["female", "male"]
    
    data = tf.data.Dataset.from_tensor_slices(image_paths)
    data = data.map(process_image).batch(16)
    
    # Transform predictions into {imagename: class} dictionary
    y_pred = model.predict(data).squeeze()
    predicted_classes = [classes[tf.math.round(pred).numpy().astype(int)] 
                         for pred in y_pred]
    predicted_images = dict(zip(filenames,predicted_classes))
    return predicted_images 
   
    
def main():
    import importlib
    import json
    import sys
    globals()["os"] = importlib.import_module("os")
    # Silencing tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Importing additional libraries
    
    libraies = ["tensorflow"]
    for library, lib_name in zip(libraies, ["tf"]):
        try:
            globals()[lib_name] = importlib.import_module(library)
        except:
            raise Exception("Failed to import {}, check packages".format(library))
    if tf.__version__ != "2.3.0":
        print("TF version is not 2.3.0, behavior may not be correct")
    
    # Taking path argument            
    try:
        path_to_images = sys.argv[1]
    except:
        path_to_images = "."
        print("Path to images are not provided, looking in current folder")
    
    # Preventing memory errors with GPU (copied from TF documentation)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)
    
    
    #Looking for images
    image_paths, filenames = find_images(path_to_images)
    if image_paths:   
        # Loading model
        try:
            mobilev2_model = tf.keras.models.load_model('./model')
        except:
            raise Exception("Failed to load model. Must be in /model folder")
        
        # Making predictions and saving them
        predictions = make_predictions(mobilev2_model, image_paths, filenames)
        with open('process_results.json', 'w') as outfile:
            json.dump(predictions, outfile)
    else:
        print("No .jpg or .jpeg files were found")

if __name__ == "__main__":       
    main()

        