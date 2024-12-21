import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_heatmap(data, working_dir='', save_path ='', save_width=20, save_height=20, dpi=300, file_name='heatmap.png'):

    if save_path == '':
        save_path = os.path.join(working_dir, file_name + "_heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False)
    plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight') 
    plt.close()