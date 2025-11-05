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

