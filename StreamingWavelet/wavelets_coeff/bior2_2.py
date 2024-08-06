""" Biorthogonal 2.2 wavelet """


class Biorthogonal22:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior2.2/
    """
    __name__ = "Biorthogonal Wavelet 2.2"
    __motherWaveletLength__ = 6  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.0,
        -0.1767766952966369,
        0.3535533905932738,
        1.0606601717798214,
        0.3535533905932738,
        -0.1767766952966369,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.3535533905932738,
        -0.7071067811865476,
        0.3535533905932738,
        0.0,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0,
        0.3535533905932738,
        0.7071067811865476,
        0.3535533905932738,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.0,
        0.1767766952966369,
        0.3535533905932738,
        -1.0606601717798214,
        0.3535533905932738,
        0.1767766952966369,
    ]
