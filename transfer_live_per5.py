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

print(next(iter(train)))