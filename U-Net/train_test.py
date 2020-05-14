import sys

from layers import *
from model import *
from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

trainGen = trainGenerator(2,'data/membrane/train', 'image', 'label', data_gen_args)
# Test has no augmentation!
testGen = trainGenerator(2,'data/membrane/val', 'image', 'label', data_gen_args)

# Pairs of scale fact and group invariance...

tests = [(2+1, 8, IdentityGroup, "Identity"),  # Before had 1/2, 1/1
         (2+2, 8, IdentityGroup, "Identity"),  # Now from 3/8 - 10/8
         (2+3, 8, IdentityGroup, "Identity"),
         (2+4, 8, IdentityGroup, "Identity"),
         (2+5, 8, IdentityGroup, "Identity"),
         (2+6, 8, IdentityGroup, "Identity"),
         (2+7, 8, IdentityGroup, "Identity"),
         (2+8, 8, IdentityGroup, "Identity"),

         (2+1, 16, RotFlipSymGen, "RotFlips"), # Before had 1/2, 1/4, 1/8
         (2+2, 16, RotFlipSymGen, "RotFlips"),
         (2+3, 16, RotFlipSymGen, "RotFlips"),
         (2+4, 16, RotFlipSymGen, "RotFlips"),
         (2+5, 16, RotFlipSymGen, "RotFlips"),
         (2+6, 16, RotFlipSymGen, "RotFlips"),
         (2+7, 16, RotFlipSymGen, "RotFlips"),
         (2+8, 16, RotFlipSymGen, "RotFlips"),
         ]
# tests = [(1, IdentityGroup, "Identity"),
#          (2, IdentityGroup, "Identity"),
#
#          (1, HorizontalSymGen, "HorizontalFlips"),
#          (2, HorizontalSymGen, "HorizontalFlips"),
#          (4, HorizontalSymGen, "HorizontalFlips"),
#
#          (1, FlipsSymGen,      "Flips"), # 5
#          (2, FlipsSymGen,      "Flips"), # 6
#          (4, FlipsSymGen,      "Flips"),
#
#          (2, RotFlipSymGen,    "RotFlips"), # ix: 8
#          (4, RotFlipSymGen,    "RotFlips"),
#          (8, RotFlipSymGen,    "RotFlips"),
#
#          (1, None, "Baseline"), # 11
#          (2, None, "Baseline"), # 12
#          ]

#for t in tests:
tstnum = int(sys.argv[1])
lr = 1e-5
if len(sys.argv) > 2:
    lr = 1e-4

d, sf, group, name = tests[tstnum]

if group != None:
    model = unet_sym(None, (256,256,1), d, sf, lr, group)
else:
    model = unet(None, (256,256,1), sf, lr)

# Number of weights in model
trainable_count = int(
    np.sum([K.count_params(p) for p in list(model.trainable_weights)]))

# TODO: Make this less ugly
mod_name = "UnetSym-" + name + "-sf" + str(d) + "/" + str(sf) + "-weights" + str(trainable_count) + "-lr" + str(lr)
print("Now training:", mod_name)

# Callback to log to tensorboard...
log_dir = 'logs/scalars/' + mod_name
tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
print("!!!!!!!!!!!!! Logging to", log_dir, "!!!!!!!!!!!!!!!!!!")

#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(trainGen,
                    validation_data=testGen,
                    validation_steps=50,
                    steps_per_epoch=300,
                    epochs=30,
                    callbacks=[tf_callback],
                    verbose=0)




print(trainable_count)
