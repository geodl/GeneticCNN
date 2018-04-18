from keras import Sequential, callbacks
from keras.layers import *
from keras.losses import categorical_crossentropy
from evo_utils import *

from data import prepare_data
from model import EvoModel

epochs = 10
batch_size = 32
output_classes = 10


(x_train, y_train), (x_test, y_test) = prepare_data()

model = EvoModel(getRandomProps(100))
model.fit(x_train, y_train)
score = model.evaluate(x_test, y_test)
print(score)

