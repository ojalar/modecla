from PIL import Image, ImageFilter
import numpy as np

# This is the implementation of the random blur image augmentation method

class RandomBlur(object):
    def __init__(self, p, bounds):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.prob = p

    def __call__(self, img):
        if np.random.uniform(0,1) < self.prob:
            sigma = np.random.randint(self.lower_bound, self.upper_bound)
            img = img.filter(ImageFilter.GaussianBlur(radius = sigma))

        return img
