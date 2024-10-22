import tensorflow as tf
from tqdm.keras import TqdmCallback
import json
import pandas as pd

# training preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\matth\VSC\Projects\foodRecog\archive\train',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (64, 64),
    shuffle = True,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False,
    verbose = True
)

# validation preprocessing
validation_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\matth\VSC\Projects\foodRecog\archive\validation',
    labels = "inferred",
    label_mode = "categorical",
    class_names = None,
    color_mode = "rgb",
    batch_size = 32,
    image_size = (64, 64),
    shuffle = True,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = "bilinear",
    follow_links = False,
    crop_to_aspect_ratio = False,
    pad_to_aspect_ratio = False,
    data_format = None,
    verbose = True,
)

def build(): # building model check README.md
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation='relu', input_shape=[64, 64, 3]))
    # cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu', input_shape=[64, 64, 3]))
    # cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation='relu'))
    # cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
    cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
    cnn.add(tf.keras.layers.Dense(units = 256, activation = 'relu'))
    cnn.add(tf.keras.layers.Dropout(0.5)) # avoid overfitting

    cnn.add(tf.keras.layers.Dense(units = 36, activation = 'softmax'))

    # compiling and training
    cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy']) # optimizer = 'adam'
    training_hist = cnn.fit(
        x=training_set, 
        validation_data = validation_set, 
        epochs = 30, # 32 epochs
        callbacks = [TqdmCallback(verbose=1)]
    )
    
    cnn.save('trained_model.h5')
    history_dict = training_hist.history # stores in loss, accuracy, val_loss, val_accuracy
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f)

    history_df = pd.DataFrame(history_dict)
    history_df.to_csv('training_history.csv', index=False)

build()
