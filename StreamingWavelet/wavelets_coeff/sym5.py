""" Symlet 5 wavelet """


class Symlet5:
    """
    Properties
    ----------
     near symmetric, orthogonal, biorthogonal

    All values are from http://wavelets.pybytes.com/wavelet/sym5/
    """
    __name__ = "Symlet Wavelet 5"
    __motherWaveletLength__ = 10  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        0.027333068345077982,
        0.029519490925774643,
        -0.039134249302383094,
        0.1993975339773936,
        0.7234076904024206,
        0.6339789634582119,
        0.01660210576452232,
        -0.17532808990845047,
        -0.021101834024758855,
        0.019538882735286728,
    ]

    # high-pass
    decompositionHighFilter = [
        -0.019538882735286728,
        -0.021101834024758855,
        0.17532808990845047,
        0.01660210576452232,
        -0.6339789634582119,
        0.7234076904024206,
        -0.1993975339773936,
        -0.039134249302383094,
        -0.029519490925774643,
        0.027333068345077982,
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.019538882735286728,
        -0.021101834024758855,
        -0.17532808990845047,
        0.01660210576452232,
        0.6339789634582119,
        0.7234076904024206,
        0.1993975339773936,
        -0.039134249302383094,
        0.029519490925774643,
        0.027333068345077982,
    ]

    # high-pass
    reconstructionHighFilter = [
        0.027333068345077982,
        -0.029519490925774643,
        -0.039134249302383094,
        -0.1993975339773936,
        0.7234076904024206,
        -0.6339789634582119,
        0.01660210576452232,
        0.17532808990845047,
        -0.021101834024758855,
        -0.019538882735286728,
    ]
