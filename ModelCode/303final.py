import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
import pathlib
data_dir = pathlib.Path("fish_data")
image_count = len(list(data_dir.glob('*.png')))



#Batch and Image Size#
batch_size = 10
img_height = 445
img_width = 590

#Train and Validate Set#
train_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="training",
	seed=770,
	image_size=(img_height, img_width),
	batch_size=batch_size)
	
val_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="validation",
	seed=770,
	image_size=(img_height, img_width),
	batch_size=batch_size)

#Class Names#
class_names = train_ds.class_names

#Overlaps and Preprocessing#
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

#Build Model#
model = Sequential([
	#adjust layers
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Conv2D(128, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

#Checkpoint Callback#
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/checkpt", save_weights_only=True, verbose=1)

#Fit Model#
epochs=5
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs,
	callbacks=[cp_callback]
)

#Export Model#
model.save('fishmodel.h5', include_optimizer=True)










