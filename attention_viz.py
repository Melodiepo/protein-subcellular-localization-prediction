# attention_viz.py

import numpy as np
import torch as T
import matplotlib.pyplot as plt

def build_secreted_heatmap(at_dict, class_index='1', vmax=0.03, title='Secreted attention'):
    """
    Builds a big heatmap from the attention values in at_dict[class_index].
    The code slices the attention in half, etc., just like you did in your original notebook.
    
    - at_dict: dictionary from trainer.attention_dict
    - class_index: '0', '1', '2', '3' for whichever class you want
    - vmax: maximum colormap value
    - title: the title of the plot
    """

    at_secreted = T.cat(at_dict[class_index], dim=0).cpu().numpy()  # shape: [N, 2000?]
    adjust_se = np.zeros_like(at_secreted)

    for i in range(at_secreted.shape[0]):
        num_nonzero = np.count_nonzero(at_secreted[i])
        # replicate your logic:
        if num_nonzero < at_secreted.shape[1]:
            half = num_nonzero // 2
            adjust_se[i, :half] = at_secreted[i, :half]
            adjust_se[i, -(num_nonzero - half):] = at_secreted[i, half:num_nonzero]

    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(adjust_se, interpolation='nearest', aspect='auto', cmap='binary', vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Sequences Number')
    cbar = fig.colorbar(img, orientation='vertical')
    cbar.set_label('Attention Weight')
    plt.tight_layout()
    plt.show()
