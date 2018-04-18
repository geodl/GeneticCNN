import random


class Properties:
    img_rows, img_cols = (28, 28)
    output_classes = 10
    input_shape = (img_rows, img_cols, 1)
    metrics = ['accuracy']

    activation = None
    epochs = None
    batch_size = None
    dropout = None
    optimizer = None
    dense = None

    mutate_epochs_allowed=False

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
        return "epochs({}) batch_size({}) dropout({}) optimizer({}) ".format(str(self.epochs), str(self.batch_size), str(self.dropout), str(self.optimizer))

    def allow_epoch_mutation(self):
        self.mutate_epochs_allowed=True

    def breed(self, other):
        epochs = random.choice(self.epochs, other.epochs)
        batch_size= random.choice(self.batch_size, other.batch_size)
        dropout = random.choice(self.dropout, other.dropout)
        optimizer = random.choice(self.optimizer, other.optimizer)
        dense = random.choice(self.dense, other.dense)
        activation = random.choice(self.activation, other.activation)

        child = Properties(epochs, batch_size, dropout, optimizer, dense, activation)
        if self.mutate_epochs_allowed:
            child.mutate_epochs_allowed=True

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
