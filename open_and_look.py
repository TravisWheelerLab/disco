import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


if __name__ == '__main__':
    target = 'A'
    root = './test_data/spect'
    files = glob(os.path.join(root, "*"))
    files = [f for f in files if target in f]

    i = 0
    for f in files:
        if i > 5:
            break
        arr = np.load(f)
        print(arr.shape)
        plt.imshow(arr[:, 35:])
        plt.show()
        i += 1
        plt.savefig('image_offload/'+target+str(i)+'.png')