import random
import numpy as np

class ImageBuffer:
    def __init__(self, buffersize):
        self.buffer = []
        self.buffersize = buffersize
        self.index = 0

    def __len__(self):
        return len(self.buffer)

    def fetch(batch_size):
        """
        Fetch a batch size worth of images from the buffer
        """
        images = []

        if self.index + batch_size < self.buffersize:
            images = self.buffer[self.index:self.index+batch_size]
            self.index += batch_size
            return images
        else:
            images_1 = self.buffer[self.index:self.buffersize]
            images_2 = self.buffer[0:batch_size-(self.buffersize-self.index)]
            self.index = batch_size - (self.buffersize-self.index)
            return np.concatenate(images_1, images_2)

    def update(images):
        """
        Update the buffer of refined images.
        """
        if len(self.buffer) == self.buffersize:  # randomly replace images in buffer with new refined images
            indices = random.sample(range(0, self.buffersize), len(images)//2)
            image_indices = random.sample(range(0, len(images)), len(images)//2)
            self.buffer = np.put(self.buffer, indices, images[image_indices])
        else:
            if len(self.buffer) + len(images) > self.buffersize:
                self.buffer = np.concatenate(self.buffer, images[:self.buffersize-len(self.buffer)])
            else:
                self.buffer = np.concatenate(self.buffer, images)
