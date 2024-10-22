import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

cnn = tf.keras.models.load_model(r"C:\Users\matth\VSC\Projects\foodRecog\trained_model.h5")
img_path = r"C:\Users\matth\VSC\Projects\foodRecog\archive\test\apple\Image_1.jpg"
# img = cv2.imread(img_path)
# plt.imshow(img)
# plt.title("Test Image")
# plt.xticks([])
# plt.yticks([])
# plt.show(block=True)

image = tf.keras.preprocessing.image.load_img(img_path, target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) # converting image to batch
predictions = cnn.predict(input_arr)

test_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\matth\VSC\Projects\foodRecog\archive\test',
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

test_loss, test_accuracy = cnn.evaluate(test_set)
result_index = np.where(predictions[0] == max(predictions[0]))
# img = cv2.imread(img_path)
# plt.imshow(img)
# plt.title("Test Image")
# plt.xticks([])
# plt.yticks([])
# plt.show(block=True)
# file = open("labels.txt", "w")
# for i in test_set.class_names:
#     file.write(i + "\n")
# file.close()
print("It's a {}".format(test_set.class_names[result_index[0][0]]))
print(f"Our test set had an accuracy of {test_accuracy} and loss of {test_loss}!")