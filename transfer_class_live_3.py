import tensorflow as tf

# First part is all the same as transfer.py
# Skip it for now.
# set your batch size, seed, and get the data loaded.

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
    # same inputs as in function-based setup
)