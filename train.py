import os
# Silencing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
import tensorflow as tf
if tf.__version__ != "2.3.0":
    print("TF version is not 2.3.0, behavior may not be correct")

tf.get_logger().setLevel('INFO')


import tensorflow_hub as hub

def create_dataset(data_dir):  
    # Generator that performs data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range=20,
                                    width_shift_range=0.15,
                                    height_shift_range=0.15,
                                    horizontal_flip=True,
                                    zoom_range=0.15,
                                    fill_mode="constant",
                                    cval=0) # Black padding
    # Setup flow from directory
    train_generator = train_datagen.flow_from_directory(
                                    data_dir,
                                    target_size=(96, 96),
                                    batch_size=32,
                                    class_mode='binary')
    
    return train_generator
    

def warmup_scheduler(epoch, lr):
    if epoch < 20:
        return lr * 1.6
    else:
        return lr
    
def train_and_save(data):
    MODULE_HANDLE ="https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4"
    
    # Loading MobilenetV2
    base_model = hub.KerasLayer(MODULE_HANDLE, trainable=False)
    
    inputs = tf.keras.layers.Input(shape=(96, 96, 3))
    # Normalization of inputs
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Training only top layer
    print("Training first 10 epochs with freezed base model. 40 more epochs ahead")
    optimizer = tf.keras.optimizers.SGD(lr=0.05, momentum=0.9, decay=0.01)
    model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])

    freezed_history = model.fit(data,
                                epochs=10,
                                verbose=1)
    
    # Unfreezing model
    base_model.trainable = True
    
    # Changing optimizer and adding learning rate schedule
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(warmup_scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-7),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
    
    print("Unfreezing weights. Training full model for 40 epochs")
    unfreezed_history = model.fit(data,
                                  initial_epoch=10,
                                  epochs=50,
                                  callbacks=[lr_scheduler],
                                  verbose=1)
    # Saving model
    model_path = "./model"
    model.save(model_path)
    
    
    # Uniting and saving histories
    h1 = freezed_history.history
    h2 = unfreezed_history.history
    for key in h2:
        if key != "lr":
            h1[key].extend(h2[key])
    np.save('history', h1)
    
    print("Finished. Created model and history files.")
    
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
        except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)
    else:
        print("Failed to connect GPU. Training can be slow!")
    
    dataset = create_dataset(path_to_images)
    train_and_save(dataset)

if __name__ == "__main__": 
    main()
    