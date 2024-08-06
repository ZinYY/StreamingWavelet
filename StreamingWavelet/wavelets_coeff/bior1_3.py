""" Biorthogonal 1.3 wavelet """


class Biorthogonal13:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior1.3/
    """
    __name__ = "Biorthogonal Wavelet 1.3"
    __motherWaveletLength__ = 6  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.08838834764831845,
        0.08838834764831845,
        0.7071067811865476,
        0.7071067811865476,
        0.08838834764831845,
        -0.08838834764831845,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.0,
        -0.7071067811865476,
        0.7071067811865476,
        0.0,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0,
        0.0,
        0.7071067811865476,
        0.7071067811865476,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.08838834764831845,
        -0.08838834764831845,
        0.7071067811865476,
        -0.7071067811865476,
        0.08838834764831845,
        0.08838834764831845,
    ]
