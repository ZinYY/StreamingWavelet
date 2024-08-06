""" Daubechies 4 wavelet """


class Daubechies4:
    """
    Properties
    ----------
    asymmetric, orthogonal, bi-orthogonal

    All values are from http://wavelets.pybytes.com/wavelet/db4/
    """
    __name__ = "Daubechies Wavelet 4"
    __motherWaveletLength__ = 8  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        - 0.18703481171888114,
        - 0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523
    ]

    # high-pass
    decompositionHighFilter = [
        -0.23037781330885523,
        0.7148465705525415,
        - 0.6308807679295904,
        - 0.02798376941698385,
        0.18703481171888114,
        0.030841381835986965,
        - 0.032883011666982945,
        - 0.010597401784997278,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.23037781330885523,
        0.7148465705525415,
        0.6308807679295904,
        - 0.02798376941698385,
        - 0.18703481171888114,
        0.030841381835986965,
        0.032883011666982945,
        - 0.010597401784997278,
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.010597401784997278,
        - 0.032883011666982945,
        0.030841381835986965,
        0.18703481171888114,
        - 0.02798376941698385,
        - 0.6308807679295904,
        0.7148465705525415,
        - 0.23037781330885523,
    ]
