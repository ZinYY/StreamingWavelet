""" Daubechies 2 wavelet """


class Daubechies2:
    """
    Properties
    ----------
    asymmetric, orthogonal, bi-orthogonal

    All values are from http://wavelets.pybytes.com/wavelet/db2/
    """
    __name__ = "Daubechies Wavelet 2"
    __motherWaveletLength__ = 4  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025
    ]

    # high-pass
    decompositionHighFilter = [
        -0.48296291314469025,
        0.836516303737469,
        - 0.22414386804185735,
        - 0.12940952255092145,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.48296291314469025,
        0.836516303737469,
        0.22414386804185735,
        - 0.12940952255092145
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.12940952255092145,
        - 0.22414386804185735,
        0.836516303737469,
        - 0.48296291314469025
    ]
