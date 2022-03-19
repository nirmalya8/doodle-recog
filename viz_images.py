import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_grid():
    figure(figsize=(20, 20), dpi=80)
    fig, axs = plt.subplots(5, 2, constrained_layout=True)
    g1,g2 = [0,0]
    base_path = 'Data'
    for i in os.listdir(base_path):
        full_file = np.load(os.path.join(base_path,i))
        one_sample = full_file[0,:]
        print(one_sample.shape)
        print(type(one_sample))
        one_sample = np.reshape(one_sample,(28,28))
        print(one_sample.shape)
        axs[g1, g2].imshow(one_sample,cmap='gray')
        axs[g1, g2].set_title(i)
        g1+=1
        if(g1 == 5):
            g1 = 0
            g2+= 1

    plt.show()

if __name__ == '__main__':
    plot_grid()