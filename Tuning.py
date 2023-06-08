# Hyperparameter Tuning for VGG16 and ResNet50 Models

from keras_tuner.tuners import RandomSearch
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

# RESNET50

# Tune for Learning Rate in Adam Optimizer
def build_model(hp):
    resnet_model = ResNet50(weights=None, include_top=False)
    resnet_model.load_weights('/kaggle/input/preweights/resnet50.weights')
    model = keras.models.Sequential()
    model.add(resnet_model)
    
    # Train all layers
    for layer in resnet_model.layers:
        layer.trainable = True
    
    
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5)
    
    model.add(BatchNormalization())
    model.add(layers.Dense(524, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5)
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='LOG')),
                  metrics=['acc'a])
    return model

# Define a callback for early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_acc',
    max_trials=7,
    executions_per_trial=1,
    directory='/kaggle/working/',
    project_name='resnet_hyperparameter_tuning',
    overwrite=True
)

# Search for the best hyperparameters
tuner.search(train_data,
             epochs=7,
             validation_data=validation_data,
             callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}. The optimal dropout rate is {best_hps.get('dropout_rate')}.
""")

# Tune for Drop-out Probability
def build_model(hp):
    resnet_model = ResNet50(weights=None, include_top=False)
    resnet_model.load_weights('/kaggle/input/preweights/resnet50.weights')
    model = keras.models.Sequential()
    model.add(resnet_model)
    
    # Train all layers
    for layer in resnet_model.layers:
        layer.trainable = True
    
    
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))    
    model.add(BatchNormalization())
    model.add(layers.Dense(524, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='LOG')),
                  metrics=['acc'a])
    return model

# Define a callback for early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_acc',
    max_trials=7,
    executions_per_trial=1,
    directory='/kaggle/working/',
    project_name='resnet_hyperparameter_tuning',
    overwrite=True
)

# Search for the best hyperparameters
tuner.search(train_data,
             epochs=7,
             validation_data=validation_data,
             callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}. The optimal dropout rate is {best_hps.get('dropout_rate')}.
""")

# VGG16

def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Train all layers
    for layer in base_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=hp.Int('dense_1_units', min_value=256, max_value=512, step=32),
                    activation='relu',
                    kernel_regularizer=l2(0.01)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(BatchNormalization())
    model.add(Dense(units=hp.Int('dense_2_units', min_value=128, max_value=256, step=32),
                    activation='relu',
                    kernel_regularizer=l2(0.01)))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # how many model variations to test?
    executions_per_trial=3,  # how many trials per variation?
    directory='random_search',
    project_name='VGG16')

tuner.search(train_data,
             epochs=10,
             validation_data=validation_data)
