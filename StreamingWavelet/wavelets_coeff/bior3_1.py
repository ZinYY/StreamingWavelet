""" Biorthogonal 3.1 wavelet """


class Biorthogonal31:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior3.1/
    """
    __name__ = "Biorthogonal Wavelet 3.1"
    __motherWaveletLength__ = 4  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.3535533905932738,
        1.0606601717798214,
        1.0606601717798214,
        -0.3535533905932738,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.1767766952966369,
        0.5303300858899107,
        -0.5303300858899107,
        0.1767766952966369,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.1767766952966369,
        0.5303300858899107,
        0.5303300858899107,
        0.1767766952966369,
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.3535533905932738,
        -1.0606601717798214,
        1.0606601717798214,
        0.3535533905932738,
    ]
