""" Biorthogonal 1.5 wavelet """


class Biorthogonal15:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior1.5/
    """
    __name__ = "Biorthogonal Wavelet 1.5"
    __motherWaveletLength__ = 10  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.01657281518405971,
        -0.01657281518405971,
        -0.12153397801643787,
        0.12153397801643787,
        0.7071067811865476,
        0.7071067811865476,
        0.12153397801643787,
        -0.12153397801643787,
        -0.01657281518405971,
        0.01657281518405971,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.0,
        0.0,
        0.0,
        -0.7071067811865476,
        0.7071067811865476,
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
        0.7071067811865476,
        0.7071067811865476,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.01657281518405971,
        0.01657281518405971,
        -0.12153397801643787,
        -0.12153397801643787,
        0.7071067811865476,
        -0.7071067811865476,
        0.12153397801643787,
        0.12153397801643787,
        -0.01657281518405971,
        -0.01657281518405971,
    ]
