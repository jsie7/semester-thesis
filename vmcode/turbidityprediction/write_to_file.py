#!/usr/bin/env python

"""Module containing the function to write predictions to eaf files.

"""

# built-in modules
import os

# third party modules
import numpy as np
import pandas as pd

# project specific modules
import vmcode.recproc.eaf_processing as eafp

FPS = 40
"""int: Number of frames per second in the video files.

"""


def write_to_file(data, base_path=None):
    """Write to file function.

    Write a dict of predictions to separate files. For each key (file name) the
    values are written to the corresponding eaf file.
    The function returns a dict with None values to adhere to the data pipeline
    convention.

    Args:
        data (dict): Dictionary containing the predicted turbidities. Keys are
            the video file names, values are tuples (y_true, y_predicted).
            The tuple is of the format, which is returned by the turbidity
            prediction function.
            WARNING: If this function is used within a data pipeline, this
              dictionary will be produced by the pipeline. This could be
              a source of error.
        base_path (str): Path to the base folder of the video files.

    Returns:
        dictionary: Keys are the video file names, values are None.

    """
    return_data = {}
    for file in data:
        y_true = data[file][0]
        y_proba = data[file][1]
        video = file.split('/')[0]
        return_data[file] = None
        eaf_file_found = False

        if os.path.isdir(base_path + video):
            for f in os.listdir(base_path + video):
                if f.endswith('.eaf'):
                    eaf_file_found = True
                    v = np.round(y_proba[:, 1], decimals=3)
                    v_compressed = np.concatenate([v[0].reshape(-1),
                        v[np.where(v[:-1] != v[1:])[0] + 1],
                        v[-1].reshape(-1)], axis=0)
                    ind_compressed = np.concatenate([np.zeros(1), np.where(
                        v[:-1] != v[1:])[0] + 1, np.array(v.shape[0])
                        .reshape(-1)], axis=0)
                    t_compressed = ind_compressed / FPS * 1000
                    name = np.repeat(np.array(["Turbidity_pred"]),
                        ind_compressed.shape[0])
                    changepoints = pd.DataFrame(data={"column": name,
                                                      "value": v_compressed,
                                                      "time": t_compressed
                                                              .astype(int)})
                    eafp.write_to_eaf(changepoints, base_path + video + "/" +
                                      video + ".eaf", base_path + video + "/" +
                                      video + "_new.eaf")
            if not eaf_file_found:
                raise FileNotFoundError("No .eaf file found in {0}!".format(
                    base_path + video))
                return
        else:
            raise FileNotFoundError("Directory {0} does not exist!".format(
                base_path + video))
            return

    return return_data
