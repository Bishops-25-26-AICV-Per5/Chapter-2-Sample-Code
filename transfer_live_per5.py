import tensorflow as tf

BATCH_SIZE = 4

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

# All pretrained models come with a preprocess_input() function. Always
#   use that on both train and validation Datasets.
train = train.map(lambda x, y: (
        tf.keras.applications.resnet50.preprocess_input(x), y))
validation = validation.map(lambda x, y: (
        tf.keras.applications.resnet50.preprocess_input(x), y))
# This is supposed to speed things up.
# .cache() means that tensorflow will keep the Dataset in memory after 
#   it is loaded for the first time.
# .prefetch() will allow tensorflow to load the next batch while the 
#   current batch is being trained.
train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

res = tf.keras.applications.ResNet50(
    include_top = False,
    input_shape = (224, 224, 3),
    pooling = None,
)
res.trainable = False

inputs = tf.keras.Input(shape = (224, 224, 3))
outputs = res(inputs)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
loss = tf.keras.losses.CategoricalCrossentropy()

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = ["accuracy"],
)

print(model(next(iter(train))[0]))