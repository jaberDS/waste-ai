import tensorflow as tf
from tensorflow.keras import layers, models

# 📁 Dataset
data_dir = "data"
img_size = (160, 160)
batch_size = 32
seed = 123

# 🟢 Train / Validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# 🔄 Normalize
normalization = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 🔀 Data augmentation (only applied during training)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# 🧠 BASE MODEL (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze pretrained layers

# 🧠 NEW MODEL
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation,           # augment training images
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),         # slightly higher dropout
    layers.Dense(len(class_names), activation='softmax')
])

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🚀 Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 💾 Save
model.save("waste_model.keras")
print("✅ Transfer learning model saved!")
