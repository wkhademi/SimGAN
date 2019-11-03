import os
import numpy as np
from PIL import Image


def load_data(path):
    images = []

    im = Image.open(path)
    print('%d images'%im.n_frames)

    step = im.n_frames//1000
    n = 0
    for i in range(1000):
        if n >= im.n_frames:
            raise ValueError('trying to access frame %d/%d (step:%d)'%(n,im.n_frames,step))
        im.seek(n)
        n += step
        images.append(im.crop())

    images = np.stack(images)

    return images


def random_crop_generator(data, crop_size, batch_size):
    while True:
        inds = np.random.randint(data.shape[0], size=batch_size)
        y = np.random.randint(data.shape[1]-crop_size, size=batch_size)
        x = np.random.randint(data.shape[2]-crop_size, size=batch_size)
        batch = np.zeros((batch_size, crop_size, crop_size, 1), dtype=data.dtype)
        for i,ind in enumerate(inds):
            batch[i] = data[ind,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        yield batch, None


def center_crop_generator(data, crop_size, batch_size):
    n = 0
    y = data.shape[1]//2-crop_size//2
    x = data.shape[2]//2-crop_size//2
    while True:
        for n in range(0,len(data),batch_size):
            batch = data[n:n+batch_size,y:y+crop_size,x:x+crop_size]
            yield batch, None


def augment_images(images):
    augmented = np.concatenate((images,
                              np.rot90(images, k=1, axes=(1, 2)),
                              np.rot90(images, k=2, axes=(1, 2)),
                              np.rot90(images, k=3, axes=(1, 2))))
    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented
