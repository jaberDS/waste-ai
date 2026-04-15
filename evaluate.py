import tensorflow as tf

img_size = (128, 128)
batch_size = 32

data_dir = "data"

# load dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# normalize
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# load model
model = tf.keras.models.load_model("waste_model.h5")

# evaluate
loss, acc = model.evaluate(test_ds)

print("Test Accuracy:", acc)
