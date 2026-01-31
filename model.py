import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_cnn(input_shape=(128, 128, 3), num_classes=4):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model