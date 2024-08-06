""" Biorthogonal 5.5 wavelet """


class Biorthogonal55:
    """
    Properties
    ----------
     near symmetric, not orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/bior5.5/
    """
    __name__ = "Biorthogonal Wavelet 5.5"
    __motherWaveletLength__ = 12  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.0,
        0.0,
        0.03968708834740544,
        0.007948108637240322,
        -0.05446378846823691,
        0.34560528195603346,
        0.7366601814282105,
        0.34560528195603346,
        -0.05446378846823691,
        0.007948108637240322,
        0.03968708834740544,
        0.0,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.013456709459118716,
        -0.002694966880111507,
        0.13670658466432914,
        -0.09350469740093886,
        -0.47680326579848425,
        0.8995061097486484,
        -0.47680326579848425,
        -0.09350469740093886,
        0.13670658466432914,
        -0.002694966880111507,
        -0.013456709459118716,
        0.0,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.013456709459118716,
        -0.002694966880111507,
        -0.13670658466432914,
        -0.09350469740093886,
        0.47680326579848425,
        0.8995061097486484,
        0.47680326579848425,
        -0.09350469740093886,
        -0.13670658466432914,
        -0.002694966880111507,
        0.013456709459118716,
        0.0,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.0,
        0.0,
        0.03968708834740544,
        -0.007948108637240322,
        -0.05446378846823691,
        -0.34560528195603346,
        0.7366601814282105,
        -0.34560528195603346,
        -0.05446378846823691,
        -0.007948108637240322,
        0.03968708834740544,
        0.0,
    ]
