""" Symlet 3 wavelet """


class Symlet3:
    """
    Properties
    ----------
     near symmetric, orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/sym3/
    """
    __name__ = "Symlet Wavelet 3"
    __motherWaveletLength__ = 6  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.035226291882100656,
        -0.08544127388224149,
        -0.13501102001039084,
        0.4598775021193313,
        0.8068915093133388,
        0.3326705529509569,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.3326705529509569,
        0.8068915093133388,
        -0.4598775021193313,
        -0.13501102001039084,
        0.08544127388224149,
        0.035226291882100656,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.3326705529509569,
        0.8068915093133388,
        0.4598775021193313,
        -0.13501102001039084,
        -0.08544127388224149,
        0.035226291882100656,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.035226291882100656,
        0.08544127388224149,
        -0.13501102001039084,
        -0.4598775021193313,
        0.8068915093133388,
        -0.3326705529509569,
    ]
