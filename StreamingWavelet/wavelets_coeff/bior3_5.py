""" Biorthogonal 3.5 wavelet """


class Biorthogonal35:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior3.5/
    """
    __name__ = "Biorthogonal Wavelet 3.5"
    __motherWaveletLength__ = 12  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -0.013810679320049757,
        0.04143203796014927,
        0.052480581416189075,
        -0.26792717880896527,
        -0.07181553246425874,
        0.966747552403483,
        0.966747552403483,
        -0.07181553246425874,
        -0.26792717880896527,
        0.052480581416189075,
        0.04143203796014927,
        -0.013810679320049757,
    ]

    # high-pass
    decompositionHighFilter = [
        0.0,
        0.0,
        0.0,
        0.0,
        -0.1767766952966369,
        0.5303300858899107,
        -0.5303300858899107,
        0.1767766952966369,
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
        0.1767766952966369,
        0.5303300858899107,
        0.5303300858899107,
        0.1767766952966369,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        -0.013810679320049757,
        -0.04143203796014927,
        0.052480581416189075,
        0.26792717880896527,
        -0.07181553246425874,
        -0.966747552403483,
        0.966747552403483,
        0.07181553246425874,
        -0.26792717880896527,
        -0.052480581416189075,
        0.04143203796014927,
        0.013810679320049757,
    ]
