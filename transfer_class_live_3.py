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

train = train.map(lambda x, y: 
        (tf.keras.applications.resnet50.preprocess_input(x), y))
validation = validation.map(lambda x, y:
        (tf.keras.applications.resnet50.preprocess_input(x), y))
train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

class Model:
    def __init__(self, input_shape):
        # In place of tf.keras.Model() from function-based version
        self.model = tf.keras.Sequential()
        self.res = tf.keras.applications.resnet50.ResNet50(
            include_top = False,
            input_shape = input_shape,
            pooling = "avg",
        )
        self.res.trainable = False
        self.model.add(self.res)
        # Flatten is optional if pooling is not None.
        self.model.add(tf.keras.layers.Flatten())
        # Size of the Dense layer is equal to number of classes in dataset.
        self.model.add(tf.keras.layers.Dense(5, activation="softmax"))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = ['accuracy']
        )

model = Model((224, 224, 3))

# Notice that the model is an attribute of the model instance, 
#   so use model.model....

model.model.summary()

model.model.fit(
    train,
    epochs = 50,
    verbose = 1, # If writing to screen, use 1.  To file, use 2.
    validation_data = validation,
)