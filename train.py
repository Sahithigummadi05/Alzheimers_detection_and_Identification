from data_loader import load_and_preprocess_oasis
from model_builder import build_transfer_cnn
from sklearn.model_selection import train_test_split
import tensorflow as tf

X, y = load_and_preprocess_oasis(n_subjects=200)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model, base_model = build_transfer_cnn()

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,
    callbacks=[early_stop]
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16,
    callbacks=[early_stop]
)

model.save("alzheimer_cnn_model.h5")