


import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import h5py

#=======================================================================================================================
""" 
GPU Memory growth
"""
#=======================================================================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Concatenate, AveragePooling2D,\
    UpSampling2D, Dropout, Lambda


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)

    except RuntimeError as e:
        print(e)

#=======================================================================================================================
"""
Model inherit
"""
#=======================================================================================================================
def unet(input_size=(640, 960, 3)):
    inputs = keras.Input(input_size)
    inputs_norm = Lambda(lambda x: x / 127.5 - 1.)
    conv1 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', name='down1a')(inputs)
    conv1 = Conv2D(8,  kernel_size=(3, 3),  activation='relu', padding='same', name='down1b')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(16,  kernel_size=(3, 3),  activation='relu', padding='same', name='down2a')(pool1)
    conv2 = Conv2D(16,  kernel_size=(3, 3),  activation='relu', padding='same', name='down2b')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(32,  kernel_size=(3, 3),  activation='relu', padding='same', name='down3a')(pool2)
    conv3 = Conv2D(32,  kernel_size=(3, 3),  activation='relu', padding='same', name='down3b')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(64,  kernel_size=(3, 3),  activation='relu', padding='same', name='down4a')(pool3)
    conv4 = Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same', name='down4b')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(128,  kernel_size=(3, 3),  activation='relu', padding='same', name='upreg1')(pool4)
    conv5 = Conv2D(128,  kernel_size=(3, 3),  activation='relu', padding='same', name='upreg2')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(64,  kernel_size=(3, 3),  activation='relu', padding='same', name='up1a')(up6)
    conv6 = Conv2D(64,  kernel_size=(3, 3),  activation='relu', padding='same', name='up1b')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(32,  kernel_size=(3, 3),  activation='relu', padding='same', name='up2a')(up7)
    conv7 = Conv2D(32,  kernel_size=(3, 3),  activation='relu', padding='same', name='up2b')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(16,  kernel_size=(3, 3),  activation='relu', padding='same', name='up3a')(up8)
    conv8 = Conv2D(16,  kernel_size=(3, 3),  activation='relu', padding='same', name='up3b')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(8,  kernel_size=(3, 3),  activation='relu', padding='same', name='up4a')(up9)
    conv9 = Conv2D(8,  kernel_size=(3, 3),  activation='relu', padding='same', name='up4b')(conv9)

    conv10 = Conv2D(1,  kernel_size=(1, 1),  activation='sigmoid', name='upreg3')(conv9)

    model = keras.Model(inputs, conv10)
    return model

def dice_loss(y_true, y_pred):
    y_true_f = keras.layers.Flatten()(y_true)
    y_pred_f = keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return 2 * (intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)

def acc(y_true, y_pred):
    return -dice_loss(y_true, y_pred)


heights = 640
width = 960

model = unet()
optimizer = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, decay=0)
model.compile(optimizer=optimizer,
              loss=dice_loss, metrics=[acc])

model.summary()

#=======================================================================================================================
""" 
Read Dataset from repository. Place both repository in the same path of this file.
"""
#=======================================================================================================================

dir_label = ['object-dataset',
            'object-detection-crowdai']

df_files1 = pd.read_csv(dir_label[1]+'/labels.csv', header=0)
df_vehicles1 = df_files1[(df_files1['Label']=='Car') | (df_files1['Label']=='Truck')].reset_index()
df_vehicles1 = df_vehicles1.drop(labels='index',axis= 1)
df_vehicles1['File_Path'] =  dir_label[1] + '/' +df_vehicles1['Frame']
df_vehicles1 = df_vehicles1.drop(labels='Preview URL', axis=1)
df_vehicles1 =df_vehicles1[['File_Path','Frame','Label','ymin','xmin','ymax','xmax']]

column= ['Frame',  'xmin', 'xmax', 'ymin','ymax', 'ind', 'Label','RM']
df_files2 = pd.read_csv('object-dataset/labels.csv', header=None,  delimiter=r"\s+", names=column)
df_vehicles2 = df_files2[(df_files2['Label']=='car') | (df_files2['Label']=='truck')].reset_index()
df_vehicles2 = df_vehicles2.drop(labels= 'index', axis= 1)
df_vehicles2 = df_vehicles2.drop(labels= 'RM', axis= 1)
df_vehicles2 = df_vehicles2.drop(labels= 'ind', axis= 1)
df_vehicles2['File_Path'] = dir_label[0] + '/' +df_vehicles2['Frame']
df_vehicles2 =df_vehicles2[['File_Path','Frame','Label','ymin','xmin','ymax','xmax']]

df_vehicles = pd.concat([df_vehicles1,df_vehicles2]).reset_index()
df_vehicles = df_vehicles.drop(labels= 'index', axis=1)

#=======================================================================================================================
""" 
Data Augmentation consist translation, and brightness
"""
#=======================================================================================================================
def brightness(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def translation(img, boundary_f, range):
    boundary_f = boundary_f.copy(deep=True)
    tr_x = range * np.random.uniform() - range / 2
    tr_y = range * np.random.uniform() - range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    rows, cols, channels = img.shape
    boundary_f['xmin'] = boundary_f['xmin'] + tr_x
    boundary_f['xmax'] = boundary_f['xmax'] + tr_x
    boundary_f['ymin'] = boundary_f['ymin'] + tr_y
    boundary_f['ymax'] = boundary_f['ymax'] + tr_y

    new_img = cv2.warpAffine(img, Trans_M, (cols, rows))
    return new_img, boundary_f


def stretching(img, boundary_f, scale_range):
    boundary_f = boundary_f.copy(deep=True)
    tr_x1 = scale_range * np.random.uniform()
    tr_y1 = scale_range * np.random.uniform()
    p1 = (tr_x1, tr_y1)
    tr_x2 = scale_range * np.random.uniform()
    tr_y2 = scale_range * np.random.uniform()
    p2 = (img.shape[1] - tr_x2, tr_y1)

    p3 = (img.shape[1] - tr_x2, img.shape[0] - tr_y2)
    p4 = (tr_x1, img.shape[0] - tr_y2)

    pts1 = np.float32([[p1[0], p1[1]],
                       [p2[0], p2[1]],
                       [p3[0], p3[1]],
                       [p4[0], p4[1]]])
    pts2 = np.float32([[0, 0],
                       [img.shape[1], 0],
                       [img.shape[1], img.shape[0]],
                       [0, img.shape[0]]]
                      )
    boundary_f['xmin'] = (boundary_f['xmin'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
    boundary_f['xmax'] = (boundary_f['xmax'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
    boundary_f['ymin'] = (boundary_f['ymin'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
    boundary_f['ymax'] = (boundary_f['ymax'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
    stretching_M = cv2.getPerspectiveTransform(pts1, pts2)

    new_img = cv2.warpPerspective(img, stretching_M, (img.shape[1], img.shape[0]))
    new_img = np.array(new_img, dtype=np.uint8)
    return new_img, boundary_f


def get_image_name(df, index, size, trans_range, scale_range):
    file_name = df['File_Path'][index]
    img = cv2.imread(str(file_name))
    img_size = np.shape(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    boundary_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)

    img, bb_boxes = translation(img, boundary_boxes, trans_range)
    img, bb_boxes = stretching(img, boundary_boxes, scale_range)
    img = brightness(img)

    boundary_boxes['xmin'] = np.round(boundary_boxes['xmin'] / img_size[1] * img_size_post[1])
    boundary_boxes['xmax'] = np.round(boundary_boxes['xmax'] / img_size[1] * img_size_post[1])
    boundary_boxes['ymin'] = np.round(boundary_boxes['ymin'] / img_size[0] * img_size_post[0])
    boundary_boxes['ymax'] = np.round(boundary_boxes['ymax'] / img_size[0] * img_size_post[0])
    boundary_boxes['Area'] = (boundary_boxes['xmax'] - boundary_boxes['xmin']) * (boundary_boxes['ymax']
                                                                                  - boundary_boxes['ymin'])

    return name_str, img, bb_boxes


def get_mask(img, bb_boxes_f):
    mask = np.zeros_like(img[:, :, 0])
    for i in range(len(bb_boxes_f)):
        bb_box_i = [bb_boxes_f.iloc[i]['xmin'], bb_boxes_f.iloc[i]['ymin'],
                    bb_boxes_f.iloc[i]['xmax'], bb_boxes_f.iloc[i]['ymax']]
        mask[int(bb_box_i[1]):int(bb_box_i[3]), int(bb_box_i[0]):int(bb_box_i[2])] = 1.
        mask = np.reshape(mask, (np.shape(mask)[0], np.shape(mask)[1], 1))
    return mask

def train_batch(data, batch_size=32, heights=640, width=960):
    batch_images = np.zeros((batch_size, heights, width, 3))
    batch_masks = np.zeros((batch_size, heights, width, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            name_str,img,bb_boxes = get_image_name(df_vehicles,i_line,
                                                   size=(width, heights),
                                                   trans_range=50,
                                                   scale_range=50)
            img_mask = get_mask(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks

#=======================================================================================================================
""" 
implementation
"""
#=======================================================================================================================
train_data= train_batch(data=df_vehicles,batch_size=10)

history = model.fit(train_data, batch_size=1000, epochs=100)
model.save('unet_model.h5')





























