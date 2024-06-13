import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
batch_size=64
data_dir = "data/animals"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
        validation_split=0.2,
        rotation_range=35,
        width_shift_range=0.25,
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,validation_split=0.2)

train_generator = train_datagen.flow_from_directory(data_dir,
    target_size=(299,299),
    class_mode='categorical',
    batch_size=batch_size,
    subset = "training")
validation_generator = validation_datagen.flow_from_directory(data_dir,
    target_size=(299,299),
    class_mode='categorical',
    batch_size=batch_size,
    subset = "validation")
labels = {v: k for k, v in train_generator.class_indices.items()}
pre_trained_model = InceptionV3(input_shape=(299, 299, 3), include_top=True, weights='imagenet')
for layer in pre_trained_model.layers:
  layer.trainable = False
last_output = pre_trained_model.get_layer('mixed10').output
x = tf.keras.layers.Dense(1024, activation='relu')(last_output)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(90, activation='softmax')(x)
model = tf.keras.models.Model(pre_trained_model.input, x)
model.summary()
model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
load_model = False
if load_model:
    model = tf.keras.models.load_model('FinalModel.h5')

if not load_model:
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=10)
model.evaluate(validation_generator)
model.save('FinalModel.h5')
testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,validation_split=0)
testing_data = testing_datagen.flow_from_directory('testing_images/images',
    target_size=(299,299),
    class_mode=None,
    batch_size=1,
    subset="training")
np.argmax(model.predict(testing_data))
print(labels[np.argmax(model.predict(testing_data))])