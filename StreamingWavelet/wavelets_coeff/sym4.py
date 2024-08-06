""" Symlet 4 wavelet """


class Symlet4:
    """
    Properties
    ----------
     near symmetric, orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/sym4/
    """
    __name__ = "Symlet Wavelet 4"
    __motherWaveletLength__ = 8  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.0322231006040427,
        -0.012603967262037833,
        0.09921954357684722,
        0.29785779560527736,
        -0.8037387518059161,
        0.49761866763201545,
        0.02963552764599851,
        -0.07576571478927333,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0322231006040427,
        -0.012603967262037833,
        -0.09921954357684722,
        0.29785779560527736,
        0.8037387518059161,
        0.49761866763201545,
        -0.02963552764599851,
        -0.07576571478927333,
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.07576571478927333,
        0.02963552764599851,
        0.49761866763201545,
        -0.8037387518059161,
        0.29785779560527736,
        0.09921954357684722,
        -0.012603967262037833,
        -0.0322231006040427,
    ]
