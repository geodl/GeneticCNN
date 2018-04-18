import random
import pickle as pkl

from keras import Sequential, callbacks, utils
from keras.datasets import fashion_mnist
from keras.engine import InputLayer
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy


class Model:
    def __init__(self, props) -> None:
        print(props.dense)
        self.props = props
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=props.input_shape))
        self.model.add(BatchNormalization())
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

    def fit(self, x_train, y_train, x_test=None, y_test=None, verbose=2, epochs=None):
        if x_test is None:
            return self.model.fit(x_train, y_train,
                                  epochs=self.props.epochs if epochs is None else epochs,
                                  batch_size=self.props.batch_size,
                                  callbacks=[self.tf_cb],
                                  verbose=verbose,
                                  )
        return self.model.fit(x_train, y_train,
                              epochs=self.props.epochs if epochs is None else epochs,
                              batch_size=self.props.batch_size,
                              validation_data=(x_test, y_test),
                              callbacks=[self.tf_cb],
                              verbose=verbose,
                              )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, self.props.batch_size, verbose=0)


class Tracker:
    keep_top_n = None
    keep_loser_n = None
    internal = []
    max_size = 0
    generation = 1

    @classmethod
    def load(cls, path="./tracks/default_tracker.pkl"):
        with open(path, "rb") as f:
            return pkl.load(f)

    def save(self, path="./tracks/default_tracker.pkl"):
        derp = self
        with open(path, "wb") as f:
            pkl.dump(derp, f)

    def __init__(self, size=20, keep_n=5, keep_loser=3):
        self.max_size = size
        self.keep_top_n = keep_n
        self.keep_loser_n = keep_loser
        self.internal = []
        self._fill_with_random()

    def add_model(self, prop, score=-1):
        self.internal.append((score, prop.id(), prop))

    def _sort(self):
        self.internal.sort(key=lambda tup: tup[0], reverse=True)

    def top(self):
        self._sort()
        return self.internal[0]

    def breed_next_gen(self):
        self._sort()
        top_n = self.internal[:self.keep_top_n]
        losers = self.internal[self.keep_top_n:]
        kept_losers = []
        for i in range(self.keep_loser_n):
            lucky_loser = random.choice(losers)
            losers.remove(lucky_loser)
            kept_losers.append(lucky_loser)
        self.internal = top_n + kept_losers

        n_needed = self.max_size - len(self.internal)
        next_gen = []
        for i in range(n_needed):
            par1, par2 = random.sample(self.internal, 2)
            child = par1[2].breed(par2[2])
            next_gen.append(child)
        for child in next_gen:
            self.add_model(child)

        self.internal += next_gen
        self.generation += 1

    def unscored(self):
        for tup in self.internal:
            if tup[0] < 0:
                self.internal.remove(tup)
                yield tup[2]

    def append(self, prop):
        self.add_model(prop)

    def __str__(self):
        return str(self.internal)

    def _fill_with_random(self):
        for i in range(self.max_size):
            self.add_model(Properties.random_prop())


class Properties:
    img_rows, img_cols = (28, 28)
    output_classes = 10
    input_shape = (img_rows, img_cols, 1)
    metrics = [categorical_accuracy]

    activation = None
    epochs = None
    batch_size = None
    dropout = None
    optimizer = None
    dense = None

    mutate_epochs_allowed = False

    kernel_x = 4
    kernel_y = 4

    stride_x = 1
    stride_y = 1

    def __init__(self, epochs, batch_size, dropout, dense, optimizer="adam", activation="relu"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.dense = dense
        self.optimizer = optimizer
        self.activation = activation

    def id(self):
        return "epochs{}_batch_size{}_dropout{}_optimizer{}".format(
            str(self.epochs), str(self.batch_size), '{0:.3g}'.format(self.dropout), str(self.optimizer))

    def allow_epoch_mutation(self):
        self.mutate_epochs_allowed = True

    def breed(self, other):
        epochs = random.choice([self.epochs, other.epochs])
        batch_size = random.choice([self.batch_size, other.batch_size])
        dropout = random.choice([self.dropout, other.dropout])
        optimizer = random.choice([self.optimizer, other.optimizer])
        dense = random.choice([self.dense, other.dense])
        activation = random.choice([self.activation, other.activation])
        child = Properties(epochs, batch_size, dropout, dense, optimizer, activation)
        if self.mutate_epochs_allowed:
            child.mutate_epochs_allowed = True

        return child

    def mutate(self):
        if self.mutate_epochs_allowed:
            self.mutate_epochs()
        self.mutate_batch_size()
        self.mutate_dropout()
        self.mutate_optimizer()
        self.mutate_dense()
        self.mutate_activation()

    def mutate_dropout(self):
        dropout_step = 0.05
        dropout_min = 0.0
        dropout_max = 0.75

        mutation = self.dropout + random.choice([-1, 0, 1]) * dropout_step
        if dropout_min <= mutation <= dropout_max:
            self.dropout = mutation

    def mutate_optimizer(self):
        optimizer_opt = ["adam", "rmsprop"]
        self.optimizer = random.choice([self.optimizer] + optimizer_opt)

    def mutate_batch_size(self):
        b_step = int(self.batch_size/10) + 1
        b_min = 1
        b_max = 150
        mutation = self.batch_size + random.choice([-1, 0, 1]) * b_step
        if b_min <= mutation <= b_max:
            self.batch_size = mutation

    def mutate_dense(self):
        d_min = 0
        d_max = 300
        d_step = 10
        mutation = self.dense + random.choice([-1, 0, 1]) * d_step
        if d_min <= mutation <= d_max:
            self.dense = int(mutation)

    def mutate_activation(self):
        activation_opt = ["relu", "softmax"]
        self.activation = random.choice([self.activation] + activation_opt)

    def mutate_epochs(self):
        e_min = 1
        e_max = 50
        e_step = 15

        mutation = self.epochs + random.choice([-1, 0, 1]) * e_step
        if e_min <= mutation <= e_max:
            self.epochs = mutation

    @staticmethod
    def random_prop(n=100):
        p = Properties(10, 10, .2, 75)
        for i in range(n):
            p.mutate()
        return p


class Util:
    @staticmethod
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
