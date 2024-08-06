""" Biorthogonal 4.4 wavelet """


class Biorthogonal44:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior4.4/
    """
    __name__ = "Biorthogonal Wavelet 4.4"
    __motherWaveletLength__ = 10  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.0,
        0.03782845550726404,
        -0.023849465019556843,
        -0.11062440441843718,
        0.37740285561283066,
        0.8526986790088938,
        0.37740285561283066,
        -0.11062440441843718,
        -0.023849465019556843,
        0.03782845550726404,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        -0.06453888262869706,
        0.04068941760916406,
        0.41809227322161724,
        -0.7884856164055829,
        0.41809227322161724,
        0.04068941760916406,
        -0.06453888262869706,
        0.0,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0,
        -0.06453888262869706,
        -0.04068941760916406,
        0.41809227322161724,
        0.7884856164055829,
        0.41809227322161724,
        -0.04068941760916406,
        -0.06453888262869706,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.0,
        -0.03782845550726404,
        -0.023849465019556843,
        0.11062440441843718,
        0.37740285561283066,
        -0.8526986790088938,
        0.37740285561283066,
        0.11062440441843718,
        -0.023849465019556843,
        -0.03782845550726404,
    ]
