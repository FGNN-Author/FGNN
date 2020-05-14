# DO ML!!
import sys
from datetime import datetime

import numpy as np
#import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

from checkers_load import load_games, getData

from checkers_models import *

from sklearn.model_selection import train_test_split


# Load scale from input
sc = int(sys.argv[1])
filters = int(sys.argv[2])
sym = int(sys.argv[3])

symmod = sym == 1

print("NumLayers is", sc, "filters:", filters)

# Loggin'
a = ""
if symmod:
    a = "sym-"

name = 'fullyconv2-' + a + 'crossent-fltr' + str(filters) + '-layers' + str(sc) + '-' + datetime.now().strftime("%m%d-%H%M%S")
log_dir = 'logs/scalars/' + name
tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
print("!!!!!!!!!!!!! Logging to", log_dir, "!!!!!!!!!!!!!!!!!!")

#model = getModel(sc)

batch_s = 64
epochs = 50

print("Loading games")
games = load_games()
#games = load_games("5_checkers.txt")
# games = games[:30]

X, Y = getData(games)

# Convert to (32,1)
X2 = np.reshape(X, (-1, 32, 1))


# Convert to 8*8*1
n = X.shape[0]
X2 = np.zeros((n,8,8,1))
#print(m.shape, m[::2, ::2].shape, m[1::2, 1::2].shape, x0[::2, :].shape, x0[1::2, :].shape)
X2[:, ::2, 1::2] = X[:, ::2, :]
X2[:, 1::2, ::2] = X[:, 1::2, :]

# TEST Convert back to (8,4,1)
# X3 = np.zeros((n,8,4,1))
# print(X.shape, X2.shape)
#
# X3 = X2[:, :, 1::2]
# X3[:, 1::2, :] = X2[:, 1::2, ::2]
#print(X3.shape)
#print(np.array_equal(X, X3))

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.10, random_state=42)

if symmod:
    model = fullyConvModelSymm(sc, filters)
else:
    model = literallyFuckingStupidFullyConvModel(sc, filters)

model.fit(X_train, Y_train,
          batch_size=batch_s,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[tf_callback],
          verbose=2)

# Save anyways cause.. ye
f_name = name + '.h5'
print("Saving", f_name)
model.save(f_name)
