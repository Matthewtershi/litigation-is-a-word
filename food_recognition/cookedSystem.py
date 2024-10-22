import json
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('trained_model.h5')

with open('training_history.json','r') as f:
    training_history_json = json.load(f)

print("Validation set Accuracy: {} %".format(training_history_json['val_accuracy'][-1]*100))

training_history_df = pd.read_csv('training_history.csv')

# print(training_history_json)
# print(training_history_df)

matplotlib.use('TkAgg')
plt.ion()
epochs = [i for i in range(1, 31)] # (1,33)
plt.plot(epochs, training_history_df['accuracy'], color='red')
plt.plot(epochs, training_history_df['val_accuracy'], color='blue')
plt.xlabel('# of Epochs')
plt.ylabel('Training/Validation Accuracy')
plt.title('Visualized Training and Validation Accuracy')
plt.show(block=True)