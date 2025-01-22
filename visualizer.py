import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

# Small visualizer for object detection.

def display_objects(df, n_samples, pre_label):
    """
    Display n_samples from a given dataframe. Show objects with the highest peak values.

    Parameters:
        df (pandas): dataframe with detected objects.
        n_samples (int): number of samples.
        pre_label (str): type of objects to be displayed.
    """

    grid = math.ceil(np.sqrt(n_samples))

    fig, axis = plt.subplots(grid, grid)
    axis = axis.flatten()
    df_sorted = df[df["PRE_LABEL"] == pre_label].sort_values("PEAK_VAL", ascending = False)
    length = len(df_sorted)

    for idx in range(grid**2):
        if idx <= length:
            axis[idx].imshow(df_sorted["REGION"].iloc[idx])
            axis[idx].set_xticklabels([])
            axis[idx].set_yticklabels([])
            if pre_label == "star":
                axis[idx].set_title(f"{df_sorted['INFO'].iloc[idx]}", fontsize = 9)

        else:
            fig.delaxes(axis[idx])
            
    plt.tight_layout()
    plt.show()

# Call function.
test_objs = pd.read_pickle("objects_files.pkl")
display_objects(test_objs, 10, "star")