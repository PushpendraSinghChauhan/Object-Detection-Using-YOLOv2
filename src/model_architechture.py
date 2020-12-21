# This part is implemented by me from scratch. Here I have trained only last convolution layer and for the rest layers used pretrained
# weights. 
# It took around 8.5 hours to train this model on PASCAL VOC 2012 dataset on Google Colab using GPU framework
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow.keras.backend as K
import tensorflow as tf

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)
def YOLOv2_architecture(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, trainable=False):
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

    # First Layer 
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False, trainable=trainable)(input_image)
    x = BatchNormalization(name='norm_1', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Layer
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_2', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third Layer
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_3', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Fourth Layer
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_4', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Fifth Layer
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_5', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Sixth Layer 
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_6', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Seventh Layer
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_7', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Eighth Layer
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_8', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Nineth Layer
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_9', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Tenth Layer
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_10', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Eleventh Layer 
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_11', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Twelveth Layer 
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_12', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Thirteenth Layer
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_13', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fourteenth Layer
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_14', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Fifteenth Layer
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_15', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Sixteenth Layer
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_16', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Seventeenth Layer
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_17', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Eighteenth Layer
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_18', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Nineteenth Layer 
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_19', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Twentieth Layer
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_20', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Twenty First Layer 
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False, trainable=trainable)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21', trainable=trainable)(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Twenty Second Layer
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_22', trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Twenty Third Layer 
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model 
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    return(model, true_boxes)