""" Biorthogonal 1.1 wavelet """


class Biorthogonal11:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior1.1/
    """
    __name__ = "Biorthogonal Wavelet 1.1"
    __motherWaveletLength__ = 2  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.7071067811865476,
        0.7071067811865476,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.7071067811865476,
        0.7071067811865476,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.7071067811865476,
        0.7071067811865476,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.7071067811865476,
        -0.7071067811865476,
    ]
