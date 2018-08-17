import keras
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model


def basic_cnn_model(input_shape=(24, 24, 1), num_classes=10, filter_size1=32,
                    filter_size2=64, hidden_size=500,
                    **kwargs) -> keras.models.Model:
    """Returns a basic CNN model.
    :param freeze:
    """

    cnn_input = Input(shape=input_shape)
    x = Conv2D(filter_size1, 3, activation='relu',
               padding='same')(cnn_input)
    x = MaxPooling2D()(x)
    x = Conv2D(filter_size2, 3, activation='relu',
               padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(hidden_size, activation='relu', use_bias=True)(x)
    cnn_output = Dense(num_classes, activation='softmax', use_bias=True)(x)

    model = Model(inputs=cnn_input, outputs=cnn_output)
    return model
