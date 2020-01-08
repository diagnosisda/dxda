# ETH Zurich, IBI-CIMS, Qin Wang (wang@qin.ee)
# Utils for PHM datasets

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

from scipy.io import loadmat
from glob import glob

def get_cwru_list(load, dir="./data/cwru/", mode="all"):
    """ Get file a list of cwru files for each condition under specific load

    Args:
      load: A int chosen from [0, 1, 2, 3] sepcifying the domain (beairng load)
      dir: (Optional) Root directory for cwru dataset, where all the mat files
           are.
      mode: (Optional) Mode, "all", "20%", "50%"

    Returns:
      A dictionary of list of files. For example, get_cwru_list(1)[2] provides
        us with a list of filenames under load 1 and class 2.
    """
    if mode == "20%":
        lists = {0: dir + "normal_" + str(load) + "*.mat",
                1: dir + "12k_Drive_End_IR007_" + str(load) + "_*.mat"}
    elif mode == "50%":
        lists = {0: dir + "normal_" + str(load) + "*.mat",
             1: dir + "12k_Drive_End_IR007_" + str(load) + "_*.mat",
             2: dir + "12k_Drive_End_IR014_" + str(load) + "_*.mat",
             3: dir + "12k_Drive_End_IR021_" + str(load) + "_*.mat",
             4: dir + "12k_Drive_End_B007_" + str(load) + "_*.mat",
             }
    else:
        lists = {0: dir + "normal_" + str(load) + "*.mat",
             1: dir + "12k_Drive_End_IR007_" + str(load) + "_*.mat",
             2: dir + "12k_Drive_End_IR014_" + str(load) + "_*.mat",
             3: dir + "12k_Drive_End_IR021_" + str(load) + "_*.mat",
             4: dir + "12k_Drive_End_B007_" + str(load) + "_*.mat",
             5: dir + "12k_Drive_End_B014_" + str(load) + "_*.mat",
             6: dir + "12k_Drive_End_B021_" + str(load) + "_*.mat",
             7: dir + "12k_Drive_End_OR007@6_" + str(load) + "_*.mat",
             8: dir + "12k_Drive_End_OR014@6_" + str(load) + "_*.mat",
             9: dir + "12k_Drive_End_OR021@6_" + str(load) + "_*.mat"}
    return {label: glob(lists[label]) for label in lists}


def read_cwru_mat(filename, length=1024, sample=200, scaling=False, fft=True,
    truncate=False):
    """ Read a single .mat file and preprocess it.

    Args:
      filename: A String, name of the .mat file.
      length: An Int, telling us the length of each raw sample.
      sample: An Int, Number of samples we choose uniformaly from the series.
      scaling: A boolean, scaling the features or notself.
      fft: A boolean, FFT feature extraction or not.
      truncate: A boolean(False) or an int, specifying if we are using only
        part of the signal

    Returns:
       A list of preprocessed samples from the specific file.
    """
    data = loadmat(filename)
    key = [k for k in data.keys() if "DE_time" in k][0]
    data = data[key].reshape([-1])
    assert(sample <= len(data) - length + 1)
    if "normal" in filename:
        data = data[::4]

    if truncate:
        print("filename", filename)
        print("Before Truncate:", len(data))
        if truncate: # 120000
            data = data[:truncate]
        print("After Truncate:", len(data))
    # Split one signal to samples
    data = [data[i:i + length] for i in range(0, len(data) - length + 1 , (len(data) - length)//(sample - 1) )]
    # In some cases where (len(data) - length)//(sample - 1) ) is not an
    # integer, it is possible the resulted data's length > sample
    data = data[:sample]
    if fft:
        # Symmetric, so //2
        if scaling:
            fft = lambda sig: abs(np.fft.fft(sig, norm="ortho")[:len(sig)//2])
        else:
            fft = lambda sig: abs(np.fft.fft(sig)[:len(sig)//2])
        data = [fft(x) for x in data]
    return data



def load_cwru(load, dir="./data/cwru/" , shuffle=False, length=1024, sample=200,
              scaling=False, fft=True, truncate=False, mode="all"):
    """ Load cwru to numpy arrays

    Args:
      load: An int from [0, 1, 2, 3], specifying the bearing load.
      dir: (Optional) Root directory for cwru dataset, where all the mat files
           are.
      shuffle: A boolean, shuffle data or not.
      length: An Int, telling us the length of each raw sample.
      sample: An Int, How many samples do you want uniformly from the series.
      scaling: A boolean, scaling the features or notself.
      fft: A boolean, FFT feature extraction or not.
      truncate: A boolean(False) or an int, specifying if we are using only
        part of the signal
      mode: (Optional) Mode, "all", "healthy", "fault"

    Returns:
      Two numpy arrays (data, labels).
    """
    filelists = get_cwru_list(load, dir=dir, mode=mode)
    data, labels = [], []
    for label in filelists:
        for filename in filelists[label]:
            datum = read_cwru_mat(filename, length=length, sample=sample,
                                  scaling=scaling, fft=fft, truncate=truncate)
            data.extend(datum)
            labels.extend([label] * len(datum))
    data, labels = np.array(data), np.array(labels)
    print(data.shape)
    assert(data.shape[1] == length or data.shape[1] == length//2)
    if shuffle:
        idx = np.random.permutation(len(data))
        data, labels = data[idx], labels[idx]
    data=np.expand_dims(data, axis=-1)
    return np.float32(data), labels
