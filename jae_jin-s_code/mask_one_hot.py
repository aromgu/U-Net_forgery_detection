import numpy as np


def get_one_hot_encoded_mask(self, mask_img):
    y_img = np.squeeze(mask_img, axis=2)
    one_hot_mask = np.zeros((self.image_height, self.image_width, self.n_classes))

    back = (y_img == 0) 
    object = (y_img > 0)

    one_hot_mask[:, :, 0] = np.where(back, 1, 0)
    one_hot_mask[:, :, 1] = np.where(object, 1, 0)

    return one_hot_mask  
