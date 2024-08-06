""" Coiflets 1 wavelet """
import numpy as np


class Coiflets1:
    """
    Properties
    ----------
     near symmetric, orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/coif1/
    """
    __name__ = "Coiflets Wavelet 1"
    __motherWaveletLength__ = 6  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = np.array([
        -0.01565572813546454,
        -0.0727326195128539,
        0.38486484686420286,
        0.8525720202122554,
        0.3378976624578092,
        -0.0727326195128539,
    ])

    # high-pass
    decompositionHighFilter = np.array([
        0.0727326195128539,
        0.3378976624578092,
        -0.8525720202122554,
        0.38486484686420286,
        0.0727326195128539,
        -0.01565572813546454,
    ])

    # reconstruction filters
    # low pass
    reconstructionLowFilter = np.array([
        -0.0727326195128539,
        0.3378976624578092,
        0.8525720202122554,
        0.38486484686420286,
        -0.0727326195128539,
        -0.01565572813546454,
    ])

    # high-pass
    reconstructionHighFilter = np.array([
        -0.01565572813546454,
        0.0727326195128539,
        0.38486484686420286,
        -0.8525720202122554,
        0.3378976624578092,
        0.0727326195128539,
    ])
