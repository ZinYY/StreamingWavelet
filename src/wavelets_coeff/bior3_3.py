""" Biorthogonal 3.3 wavelet """


class Biorthogonal33:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior3.3/
    """
    __name__ = "Biorthogonal Wavelet 3.3"
    __motherWaveletLength__ = 8  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.06629126073623884,
        -0.19887378220871652,
        -0.15467960838455727,
        0.9943689110435825,
        0.9943689110435825,
        -0.15467960838455727,
        -0.19887378220871652,
        0.06629126073623884,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.0,
        -0.1767766952966369,
        0.5303300858899107,
        -0.5303300858899107,
        0.1767766952966369,
        0.0,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0,
        0.0,
        0.1767766952966369,
        0.5303300858899107,
        0.5303300858899107,
        0.1767766952966369,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.06629126073623884,
        0.19887378220871652,
        -0.15467960838455727,
        -0.9943689110435825,
        0.9943689110435825,
        0.15467960838455727,
        -0.19887378220871652,
        -0.06629126073623884,
    ]
