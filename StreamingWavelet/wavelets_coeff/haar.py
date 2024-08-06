""" Haar Wavelet """


class Haar:
    """
    Properties
    ----------
    asymmetric, orthogonal, bi-orthogonal

    All values are from http://wavelets.pybytes.com/wavelet/haar/
    """
    __name__ = "Haar Wavelet"
    __motherWaveletLength__ = 2  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.7071067811865476,
        0.7071067811865476
    ]

    # high-pass
    decompositionHighFilter = [
        -0.7071067811865476,
        0.7071067811865476
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.7071067811865476,
        0.7071067811865476
    ]

    # high-pass
    reconstructionHighFilter = [
        0.7071067811865476,
        - 0.7071067811865476
    ]
