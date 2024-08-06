""" Biorthogonal 2.6 wavelet """


class Biorthogonal26:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior2.6/
    """
    __name__ = "Biorthogonal Wavelet 2.6"
    __motherWaveletLength__ = 14  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.0,
        -0.006905339660024878,
        0.013810679320049757,
        0.046956309688169176,
        -0.10772329869638811,
        -0.16987135563661201,
        0.4474660099696121,
        0.966747552403483,
        0.4474660099696121,
        -0.16987135563661201,
        -0.10772329869638811,
        0.046956309688169176,
        0.013810679320049757,
        -0.006905339660024878,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3535533905932738,
        -0.7071067811865476,
        0.3535533905932738,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3535533905932738,
        0.7071067811865476,
        0.3535533905932738,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.0,
        0.006905339660024878,
        0.013810679320049757,
        -0.046956309688169176,
        -0.10772329869638811,
        0.16987135563661201,
        0.4474660099696121,
        -0.966747552403483,
        0.4474660099696121,
        0.16987135563661201,
        -0.10772329869638811,
        -0.046956309688169176,
        0.013810679320049757,
        0.006905339660024878,
    ]
