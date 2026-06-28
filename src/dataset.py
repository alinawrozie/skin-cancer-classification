import re
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Default Augmentation Constants
ROTATION = 180.0
SHEAR = 2.0
H_ZOOM = 8.0
W_ZOOM = 8.0
H_SHIFT = 8.0
W_SHIFT = 8.0

def get_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    def get_3x3_matrix(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])
        
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_matrix([
        c1, s1, zero,
        -s1, c1, zero,
        zero, zero, one
    ])
    
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_matrix([
        one, s2, zero,
        zero, c2, zero,
        zero, zero, one
    ])

    zoom_matrix = get_3x3_matrix([
        one/height_zoom, zero, zero,
        zero, one/width_zoom, zero,
        zero, zero, one
    ])
    
    shift_matrix = get_3x3_matrix([
        one, zero, height_shift,
        zero, one, width_shift,
        zero, zero, one
    ])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

def transform(image, DIM=224):
    XDIM = DIM % 2

    rot = ROTATION * tf.random.normal([1], dtype='float32')
    shr = SHEAR * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / H_ZOOM
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / W_ZOOM
    h_shift = H_SHIFT * tf.random.normal([1], dtype='float32')
    w_shift = W_SHIFT * tf.random.normal([1], dtype='float32')

    m = get_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift)
    x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example['label']
    return image, label

def read_unlabeled_tfrecord(example, return_image_names=True):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_name = example['image_name']
    return image, image_name if return_image_names else image

def prepare_image(img, augment=True, dim=224):
    if augment:
        img = transform(img, DIM=dim)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    img = tf.image.resize(img, [dim, dim])
    return img

def count_data_items(filenames):
    total_items = 0
    for filename in filenames:
        match = re.search(r'(test|train)(\d+)\.tfrec', filename)
        if match:
            total_items += 1
        else:
            print(f"Filename does not match the pattern: {filename}")
    return total_items

def get_dataset(files, augment=False, shuffle=False, repeat=False, labeled=True, return_image_names=True, batch_size=32, dim=224, auto_tune=tf.data.experimental.AUTOTUNE):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto_tune)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=auto_tune)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names),
                    num_parallel_calls=auto_tune)

    ds = ds.map(lambda img, label: (prepare_image(img, augment=augment, dim=dim), label), num_parallel_calls=auto_tune)
    ds = ds.prefetch(auto_tune)
    
    images = []
    labels = []
    for features, label in ds:
        images.append(features.numpy())
        labels.append(label.numpy())

    return np.array(images), np.array(labels)
