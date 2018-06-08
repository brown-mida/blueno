import numpy as np
from scipy.ndimage.interpolation import zoom

a = np.random.rand(192, 192, 200)
a = zoom(a, (1, 1, 32 / 192))
print(np.shape(a))
