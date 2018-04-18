from keras import utils
from keras.datasets import fashion_mnist


def prepare_data():
    img_rows, img_cols = 28, 28
    output_classes = 10
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = utils.to_categorical(y_train, output_classes)
    y_test = utils.to_categorical(y_test, output_classes)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    return (x_train, y_train), (x_test, y_test)
