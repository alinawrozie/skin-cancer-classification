import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import tensorflow_addons as tfa

def get_default_metrics():
    return [
        tf.keras.metrics.BinaryCrossentropy(name='Cross Entropy'),
        tf.keras.metrics.F1Score(name='f1_score'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.BinaryAccuracy(name='val_acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR')
    ]

def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Builds the baseline Custom CNN model.
    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
     
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_transfer_model(base_model, learning_rate=0.001, label_smoothing=0.05, metrics=None):
    """
    Builds the transfer learning Sequential model with gelu activation in the Dense head.
    """
    if metrics is None:
        metrics = get_default_metrics()
        
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation=tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
