""" Biorthogonal 2.4 wavelet """


class Biorthogonal24:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior2.4/
    """
    __name__ = "Biorthogonal Wavelet 1.1"
    __motherWaveletLength__ = 10  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.0,
        0.03314563036811942,
        -0.06629126073623884,
        -0.1767766952966369,
        0.4198446513295126,
        0.9943689110435825,
        0.4198446513295126,
        -0.1767766952966369,
        -0.06629126073623884,
        0.03314563036811942,
    ]

    # high-pass
    decompositionHighFilter = [
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
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
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
    ]

    # high-pass
    reconstructionHighFilter = [
        0.0,
        -0.03314563036811942,
        -0.06629126073623884,
        0.1767766952966369,
        0.4198446513295126,
        -0.9943689110435825,
        0.4198446513295126,
        0.1767766952966369,
        -0.06629126073623884,
        -0.03314563036811942,
    ]
