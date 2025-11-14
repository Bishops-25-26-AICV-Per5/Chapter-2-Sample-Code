import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

BATCH_SIZE = 32

tf.keras.utils.set_random_seed(37)

train, validation = tf.keras.utils.image_dataset_from_directory(
    "defungi",
    label_mode = "categorical",
    batch_size = BATCH_SIZE,
    image_size = (224, 224),
    seed = 37,
    validation_split = 0.2, 
    subset = "both",
)

# Apply the preprocessing function to the images, but not the labels.
# A lambda function is just a one-line throwaway function we only use once.
train = train.map(lambda x, y: 
        (tf.keras.applications.resnet50.preprocess_input(x), y))
validation = validation.map(lambda x, y:
        (tf.keras.applications.resnet50.preprocess_input(x), y))
# The cache() gets tf to keep images in RAM after loading
# The prefetch() gets tf to load the next batch while the current one is
#   training.
# If your data is large, this might hurt more than it helps.
train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

# Load and apply the ResNet50 algorithm to our data.
res = tf.keras.applications.resnet50.ResNet50(
    include_top = False,
    input_shape = (224, 224, 3),    
    pooling = "max",
)
# ResNet comes pre-trained to find features.  We don't want to mess with that.
res.trainable = False

inputs = tf.keras.Input((224, 224, 3))
outputs = res(inputs)
# Need this if we don't pool in ResNet
# outputs = tf.keras.layers.Flatten()(outputs)
outputs = tf.keras.layers.Dense(5, activation="softmax")(outputs)

# Optimizer is the function that describes how the training is computed.
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
# Loss is the function that measures the distance from the prediction to
#   the actual label.
loss = tf.keras.losses.CategoricalCrossentropy()

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = ['accuracy'],
)

model.fit(
    train,
    epochs = 5,
    verbose = 1, # If writing to screen, use 1.  To file, use 2.
    validation_data = validation,
)