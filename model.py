from keras import Sequential, callbacks
from keras.backend import categorical_crossentropy
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class EvoModel:
    def __init__(self, props) -> None:
        self.props=props
        self.model = Sequential()

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=props.input_shape))

        self.model.add(Conv2D(props.output_classes,
                              padding="same",
                              strides=(props.stride_x, props.stride_y),
                              kernel_size=(props.kernel_x, props.kernel_y)))
        self.model.add(MaxPool2D(padding="same"))
        self.model.add(Flatten())
        self.model.add(Dense(props.dense))
        self.model.add(Dense(props.output_classes))
        self.tf_cb = callbacks.TensorBoard(
            log_dir='./logs/{}'.format(props.id()),
            batch_size=props.batch_size,
            write_graph=False,
        )

        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=props.optimizer,
            metrics=props.metrics,
        )

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        if x_test is None:
            return self.model.fit(x_train, y_train,
                                  epochs=self.props.epochs,
                                  batch_size=self.props.batch_size,
                                  callbacks=[self.tf_cb],
                                  )
        return self.model.fit(x_train, y_train,
                              epochs=self.props.epochs,
                              batch_size=self.props.batch_size,
                              validation_data=(x_test, y_test),
                              callbacks=[self.tf_cb],
                              )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, self.props.batch_size)
