import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

from checkers_layers import *

# TODO: Check if this is actually correct... Might only be checking
# if the entire square's magnitude is in order, rather than comparing
# specific moves.
# It (should) be correct for models outputting a flat vector at least.
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# accuracy, top 1, top 3, top 5.
metrics = ['accuracy',
           categorical_accuracy,
           top_3_accuracy,
           top_5_accuracy]


# Adjency matrix for squares based on their relations.
def getCheckersAdjMat(incSelf=True, incUp=True, incDown=True):
    adj = np.zeros((32,32))

    def cv(x,y):
        return y*4 + x

    def inBnds(x,y):
        return 0 <= x < 4 and 0 <= y < 8

    def conn(x1, y1, x2, y2):
        if inBnds(x1, y1) and inBnds(x2, y2):
            adj[cv(x1,y1), cv(x2,y2)] = 1

    for x in range(4):
        for y in range(8):
            o = (y+1) % 2
            # Connect to ourself
            if incSelf:
                conn(x,y, x,y)
            # Connect down left/right
            if incUp:
                conn(x, y, x+o-1, y+1)
                conn(x+o-1, y+1, x, y)

            if incDown:
                conn(x, y, x+o, y+1)
                conn(x+o, y+1, x, y)

    return adj



# Only dense layer acc: 0.2894 (cross entropy)
def fullyConvModelSymm(layers, filters):
    model = Sequential()

    # Add a duplicate input for each symmetry layer.
    # This is invarient to shuffling the input.
    def lift(X):
        return tf.concat([X,X], axis=-1)
    model.add(Lambda(lift, input_shape=(8,8,1), output_shape=(8,8,2)))

    model.add(Conv2DSymm(filters, (3,3), input_shape=(8,8,1), activation='relu', padding='same'))
    for i in range(layers-1):
        model.add(Conv2DSymm(filters, (3,3), activation='relu', padding='same'))

    # 2 filters = 4 layer output, for each direction of thingy.
    model.add(Conv2DSymm(2, (3,3), activation=None, padding='same'))

    # Drop from 8x8 down to 8x4.
    def checkerDrop(X):
        xs = [X[:, i, ((i+1)%2)::2, :] for i in range(8)]
        res = tf.stack(xs, axis=1)

        return res

    model.add(Lambda(checkerDrop, input_shape=(8,8,4), output_shape=(8,4,4)))
    # Should be 8 by 4 (board) by 4 deep for each move from a square.

    model.add(Flatten(input_shape=(8,4,4)))

    #model.add(Dense(128, activation='softmax'))
    model.add(Activation('softmax')) # Safe for crossentropy

    # model.add(Dense(128, activation='relu'))

    # optim = SGD()
    optim = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=metrics)
    # model.compile(loss='mse', optimizer=optim, metrics=metrics)
    model.summary()

    return model


# Only dense layer acc: 0.2894 (cross entropy)
def literallyFuckingStupidFullyConvModel(layers, filters):
    model = Sequential()

    model.add(Conv2D(filters, (3,3), activation='relu', input_shape=(8,8,1), padding='same'))
    for i in range(layers-1):
        model.add(Conv2D(filters, (3,3), activation='relu', padding='same'))

    # 4 layer output, for each direction of thingy.
    model.add(Conv2D(4, (3,3), activation=None, padding='same'))

    def checkerDrop(X):
        print('CHECKERDROP', X.shape, flush=True)

        # layerz
        #layers = []
        #for l in range(X.shape[0]):
            #layers.append(tf.stack([X[z, i, ((i+1)%2)::2, :] for i in range(8)]))

        xs = [X[:, i, ((i+1)%2)::2, :] for i in range(8)]

        res = tf.stack(xs, axis=1)

        print("SNTHSNTEHU-------------------", xs[0].shape, res.shape, flush=True)

        return res

        #X2[:, ::2, 1::2] = X[:, ::2, :]
        #X2[:, 1::2, ::2] = X[:, 1::2, :]

        #X2 = X[:, :, 1::2, :]
        #X2[:, 1::2, :] = X[:, 1::2, ::2, :]
        #return X2
    model.add(Lambda(checkerDrop, input_shape=(8,8,4), output_shape=(8,4,4)))
    # Should be 8 by 4 (board) by 4 deep for each move from a square.

    model.add(Flatten(input_shape=(8,4,4)))

    #model.add(Dense(128, activation='softmax'))
    model.add(Activation('softmax')) # Safe for crossentropy

    # model.add(Dense(128, activation='relu'))

    # optim = SGD()
    optim = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=metrics)
    # model.compile(loss='mse', optimizer=optim, metrics=metrics)
    model.summary()

    return model

# Requires input of shape (32,).
def stupidGraphModel(layers=1):

    # Connections of different types...
    idMat = getCheckersAdjMat(incSelf=True, incUp=False, incDown=False)
    id_adj = Input(tensor=K.variable(idMat), sparse=True)

    upMat = getCheckersAdjMat(incSelf=False, incUp=True, incDown=False)
    up_adj = Input(tensor=K.variable(idMat), sparse=True)

    dnMat = getCheckersAdjMat(incSelf=False, incUp=False, incDown=True)
    dn_adj = Input(tensor=K.variable(idMat), sparse=True)


    X_in = Input(shape=(32,1))


    # Should probably be done in the input but w/e.
    # TODO: Create adjacency matrix (or fancy laplacian thing).
    def addLayer(input, filters=100, activation='relu'):
        conv1 = ConstGraphConv(filters, id_adj, activation=activation)(input)
        conv2 = ConstGraphConv(filters, up_adj, activation=activation)(input)
        conv3 = ConstGraphConv(filters, dn_adj, activation=activation)(input)
        return Add()([conv1, conv2, conv3])

    conv = X_in
    for i in range(layers):
        conv = addLayer(conv, 100, 'relu')
        #conv = ConstGraphConv(100, fixed_adj, activation='relu')(conv)

    # From (32, K) -> (32, 4) ??
    # Only the identity matrix.
    # out = ConstGraphConv(4, id_adj, activation='softmax')(conv)
    out = ConstGraphConv(4, id_adj, activation=None)(conv)

    out = Flatten()(out)
    # Normalize it with softmax.
    out = Activation('softmax')(out)

    model = Model(inputs=[X_in], outputs=out)
    # optim = SGD()
    optim = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=metrics)
    # model.compile(loss='mse', optimizer=optim, metrics=metrics)
    model.summary()

    return model


# Only dense layer acc: 0.2894 (cross entropy)
def literallyFuckingStupidModel(layers, filters):
    model = Sequential()

    model.add(Conv2D(filters, (3,3), activation='relu', input_shape=(8,8,1), padding='same'))
    for i in range(layers-1):
        model.add(Conv2D(filters, (3,3), activation='relu', padding='same'))

    model.add(Flatten(input_shape=(8,8,1)))
    model.add(Dense(128, activation='softmax'))
    # model.add(Dense(128, activation='relu'))

    # optim = SGD()
    optim = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=metrics)
    # model.compile(loss='mse', optimizer=optim, metrics=metrics)
    model.summary()

    return model

# Based on go paper...
def getModelConv(layers=12, filters=128):
    model = Sequential()

    model.add(Conv2D(filters, (3,3), activation='relu', input_shape=(8,8,1), padding='same'))
    for i in range(layers):
        model.add(Conv2D(filters, (3,3), activation='relu', padding='same'))

    # 4 filters per 'square'.. Could be just 1 & flatten?
    # Take policy directly from conv net.
    model.add(Conv2D(1, (1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # optim = SGD()
    optim = Adam(0.0001)
    # model.compile(loss='categorical_crossentropy', optimizer=optim)
    model.compile(loss='mse', optimizer=optim)
    model.summary()

    return model


# Should have:
#  2.3 million parameters, 630 million connections, and 550,000 hidden units.
def getGoModel():
    model = Sequential()

    model.add(Conv2D(192, (5,5), activation='relu', input_shape=(19,19,1), padding='same'))
    for i in range(12):
        model.add(Conv2D(192, (3,3), activation='relu', padding='same'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))

    # optim = RMSprop()
    optim = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optim)
    model.summary()

    return model

def getModel(sc):
    model = Sequential()

    model.add(Conv2D(1*sc, (2,2), activation='relu', input_shape=(8,4,1), padding='same'))
    model.add(Conv2D(1*sc, (2,2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2))) # 4*2

    model.add(Conv2D(1*sc, (2,2), activation='relu', padding='same'))

    # model.add(Conv2D(2*sc, (2,2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2))) # 2*1

    # model.add(Conv2D(4*sc, (2,2), activation='relu', padding='same'))
    #model.add(Conv2D(256, (2,2), activation='relu', padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # optim = RMSprop()
    optim = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optim)
    model.summary()

    return model


def getModel(sc):
    model = Sequential()

    model.add(Conv2D(1*sc, (2,2), activation='relu', input_shape=(8,4,1), padding='same'))
    model.add(Conv2D(1*sc, (2,2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2))) # 4*2

    model.add(Conv2D(1*sc, (2,2), activation='relu', padding='same'))

    # model.add(Conv2D(2*sc, (2,2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2))) # 2*1

    # model.add(Conv2D(4*sc, (2,2), activation='relu', padding='same'))
    #model.add(Conv2D(256, (2,2), activation='relu', padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # optim = RMSprop()
    optim = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optim)
    model.summary()

    return model

# Maxpool is ruining this network's ability to generalize correctly (I think).
def getModelMaxPool(sc):
    model = Sequential()

    model.add(Conv2D(1*sc, 3, activation='relu', input_shape=(8,4,1), padding='same'))
    model.add(Conv2D(1*sc, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 4*2

    model.add(Conv2D(2*sc, (2,2), activation='relu', padding='same'))
    model.add(Conv2D(2*sc, (2,2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 2*1

    model.add(Conv2D(4*sc, (2,2), activation='relu', padding='same'))
    #model.add(Conv2D(256, (2,2), activation='relu', padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD())
    model.summary()

    return model
