""" Daubechies 5 wavelet """


class Daubechies5:
    """
    Properties
    ----------
    asymmetric, orthogonal, bi-orthogonal

    All values are from http://wavelets.pybytes.com/wavelet/db5/
    """
    __name__ = "Daubechies Wavelet 5"
    __motherWaveletLength__ = 10  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.003335725285001549,
        - 0.012580751999015526,
        - 0.006241490213011705,
        0.07757149384006515,
        - 0.03224486958502952,
        - 0.24229488706619015,
        0.13842814590110342,
        0.7243085284385744,
        0.6038292697974729,
        0.160102397974125
    ]

    # high-pass
    decompositionHighFilter = [
        -0.160102397974125,
        0.6038292697974729,
        - 0.7243085284385744,
        0.13842814590110342,
        0.24229488706619015,
        - 0.03224486958502952,
        - 0.07757149384006515,
        - 0.006241490213011705,
        0.012580751999015526,
        0.003335725285001549
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.160102397974125,
        0.6038292697974729,
        0.7243085284385744,
        0.13842814590110342,
        - 0.24229488706619015,
        - 0.03224486958502952,
        0.07757149384006515,
        - 0.006241490213011705,
        - 0.012580751999015526,
        0.003335725285001549,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.003335725285001549,
        0.012580751999015526,
        - 0.006241490213011705,
        - 0.07757149384006515,
        - 0.03224486958502952,
        0.24229488706619015,
        0.13842814590110342,
        - 0.7243085284385744,
        0.6038292697974729,
        - 0.160102397974125,
    ]
