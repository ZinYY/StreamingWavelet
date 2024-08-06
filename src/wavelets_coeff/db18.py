""" Daubechies 18 wavelet """


class Daubechies18:
    """
    Properties
    ----------
    asymmetric, orthogonal, bi-orthogonal

    All values are from http://wavelets.pybytes.com/wavelet/db18/
    """
    __name__ = "Daubechies Wavelet 18"
    __motherWaveletLength__ = 36  # length of the mother wavelet
    __transformWaveletLength__ = 2  # minimum wavelength of input signal

    # decomposition filter
    # low-pass
    decompositionLowFilter = [
        -2.507934454941929e-09,
        3.06883586303703e-08,
        - 1.1760987670250871e-07,
        - 7.691632689865049e-08,
        1.768712983622886e-06,
        - 3.3326344788769603e-06,
        - 8.520602537423464e-06,
        3.741237880730847e-05,
        - 1.535917123021341e-07,
        - 0.00019864855231101547,
        0.0002135815619103188,
        0.0006284656829644715,
        - 0.0013405962983313922,
        - 0.0011187326669886426,
        0.004943343605456594,
        0.00011863003387493042,
        - 0.013051480946517112,
        0.006262167954438661,
        0.026670705926689853,
        - 0.023733210395336858,
        - 0.04452614190225633,
        0.05705124773905827,
        0.0648872162123582,
        - 0.10675224665906288,
        - 0.09233188415030412,
        0.16708131276294505,
        0.14953397556500755,
        - 0.21648093400458224,
        - 0.2936540407357981,
        0.14722311196952223,
        0.571801654887122,
        0.5718268077650818,
        0.31467894133619284,
        0.10358846582214751,
        0.01928853172409497,
        0.0015763102184365595
    ]

    # high-pass
    decompositionHighFilter = [
        -0.0015763102184365595,
        0.01928853172409497,
        - 0.10358846582214751,
        0.31467894133619284,
        - 0.5718268077650818,
        0.571801654887122,
        - 0.14722311196952223,
        - 0.2936540407357981,
        0.21648093400458224,
        0.14953397556500755,
        - 0.16708131276294505,
        - 0.09233188415030412,
        0.10675224665906288,
        0.0648872162123582,
        - 0.05705124773905827,
        - 0.04452614190225633,
        0.023733210395336858,
        0.026670705926689853,
        - 0.006262167954438661,
        - 0.013051480946517112,
        - 0.00011863003387493042,
        0.004943343605456594,
        0.0011187326669886426,
        - 0.0013405962983313922,
        - 0.0006284656829644715,
        0.0002135815619103188,
        0.00019864855231101547,
        - 1.535917123021341e-07,
        - 3.741237880730847e-05,
        - 8.520602537423464e-06,
        3.3326344788769603e-06,
        1.768712983622886e-06,
        7.691632689865049e-08,
        - 1.1760987670250871e-07,
        - 3.06883586303703e-08,
        - 2.507934454941929e-09
    ]

    # reconstruction filters
    # low pass
    reconstructionLowFilter = [
        0.0015763102184365595,
        0.01928853172409497,
        0.10358846582214751,
        0.31467894133619284,
        0.5718268077650818,
        0.571801654887122,
        0.14722311196952223,
        - 0.2936540407357981,
        - 0.21648093400458224,
        0.14953397556500755,
        0.16708131276294505,
        - 0.09233188415030412,
        - 0.10675224665906288,
        0.0648872162123582,
        0.05705124773905827,
        - 0.04452614190225633,
        - 0.023733210395336858,
        0.026670705926689853,
        0.006262167954438661,
        - 0.013051480946517112,
        0.00011863003387493042,
        0.004943343605456594,
        - 0.0011187326669886426,
        - 0.0013405962983313922,
        0.0006284656829644715,
        0.0002135815619103188,
        - 0.00019864855231101547,
        - 1.535917123021341e-07,
        3.741237880730847e-05,
        - 8.520602537423464e-06,
        - 3.3326344788769603e-06,
        1.768712983622886e-06,
        - 7.691632689865049e-08,
        - 1.1760987670250871e-07,
        3.06883586303703e-08,
        - 2.507934454941929e-09
    ]

    # high-pass
    reconstructionHighFilter = [
        -2.507934454941929e-09,
        - 3.06883586303703e-08,
        - 1.1760987670250871e-07,
        7.691632689865049e-08,
        1.768712983622886e-06,
        3.3326344788769603e-06,
        - 8.520602537423464e-06,
        - 3.741237880730847e-05,
        - 1.535917123021341e-07,
        0.00019864855231101547,
        0.0002135815619103188,
        - 0.0006284656829644715,
        - 0.0013405962983313922,
        0.0011187326669886426,
        0.004943343605456594,
        - 0.00011863003387493042,
        - 0.013051480946517112,
        - 0.006262167954438661,
        0.026670705926689853,
        0.023733210395336858,
        - 0.04452614190225633,
        - 0.05705124773905827,
        0.0648872162123582,
        0.10675224665906288,
        - 0.09233188415030412,
        - 0.16708131276294505,
        0.14953397556500755,
        0.21648093400458224,
        - 0.2936540407357981,
        - 0.14722311196952223,
        0.571801654887122,
        - 0.5718268077650818,
        0.31467894133619284,
        - 0.10358846582214751,
        0.01928853172409497,
        - 0.0015763102184365595
    ]
