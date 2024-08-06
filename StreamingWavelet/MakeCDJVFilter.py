import numpy as np


def mirror_filt(x):
    y = -((-1) ** (np.arange(1, len(x) + 1))) * x
    return y


def MakeCDJVFilter(request, degree=2):
    '''
    MakeCDJVFilter -- Set up filters for CDJV Wavelet Transform
    
    Usage
      [a,b,c] = MakeCDJVFilter(request,degree)
      
    Inputs
      request  string: 'HighPass', 'LowPass', 'Precondition', 'Postcondition'
      degree   integer: 2 or 3 (number of vanishing moments)
      
    Outputs
      a,b,c    filter, left edge filter, right edge filter
               ('HighPass', 'LowPass')
      a        conditioning matrix ('Precondition', 'Postcondition')
      
    Description
      CDJV have developed an algorithm for wavelets on the interval which
      preserves the orthogonality, vanishing moments, smoothness, and compact
      support of Daubechies wavelets on the line.
      
      The algorithm for wavelets on the interval of CDJV involves four objects
      not present in the usual periodized algorithm: right edge filters, left
      edge filters, and pre- and post- conditioning operators.
      These objects are supplied by appropriate requests to MakeCDJVFilter.
      
    References
      Cohen, Daubechies, Jawerth and Vial, 1992.
    '''
    
    F2 = np.array([0.482962913145, 0.836516303738, 0.224143868042, - 0.129409522551])
    F3 = np.array([0.33267055295, 0.806891509311, 0.459877502118, - 0.13501102001, - 0.085441273882, 0.035226291882])
    if str(request) == str('HighPass'):
        if degree == 2:
            LEHI2 = np.array([[- 0.7965435153, 0.5463927105, - 0.2587922607, 0, 0], [0.01003722199, 0.1223510414, 0.2274281035, - 0.8366029193, 0.4830129294]])
            REHI2 = np.array([[- 0.2575129317, 0.8014229647, - 0.5398224908, 0, 0], [0.3717189691, - 0.3639069552, - 0.7175800176, 0.4010694996, 0.2315575756]])
            a = np.flip(mirror_filt(F2))
            b = LEHI2
            c = REHI2
        if degree == 3:
            LEHI3 = np.array([[0.5837810161, 0.7936188102, 0.1609551602, - 0.05884169984, 0, 0, 0, 0], [- 0.3493401755, 0.2989205708, - 0.3283012959, - 0.332263728, 0.6982497314, - 0.287879004, 0, 0], [0.001015059936, - 3.930151414e-05, - 0.03451437279, - 0.08486981368, 0.1337306925, 0.4604064313, - 0.806893234, 0.3326712638]])
            REHI3 = np.zeros((3, 8))
            REHI3[0, 0:4] = np.array([0.07221947896, - 0.4265622004, 0.8042331363, - 0.4074777277])
            REHI3[1, 0:6] = np.array([- 0.1535052177, 0.5223942253, - 0.09819804815, - 0.7678795675, 0.2985152672, 0.1230738394])
            REHI3[2, :] = np.array([0.2294775468, - 0.4451794532, - 0.2558698634, 0.001694456403, 0.7598761492, 0.1391503023, - 0.2725472621, - 0.1123675794])
            a = np.flip(mirror_filt(F3))
            b = LEHI3
            c = REHI3
    
    if str(request) == str('LowPass'):
        if degree == 2:
            LELO2 = np.array([[0.6033325147, 0.690895529, - 0.3983129985, 0, 0], [0.03751745208, 0.4573276687, 0.8500881006, 0.223820349, - 0.1292227411]])
            RELO2 = np.array([[0.8705087515, 0.4348970037, 0.2303890399, 0, 0], [- 0.1942333944, 0.1901514021, 0.3749553135, 0.767556688, 0.4431490452]])
            a = F2
            b = LELO2
            c = RELO2
        if degree == 3:
            LELO3 = np.zeros((3, 8))
            LELO3[0, 0:4] = np.array([0.388899673, - 0.08820780195, - 0.8478413443, 0.3494874575])
            LELO3[1, 0:6] = np.array([- 0.6211483347, 0.5225274354, - 0.2000079353, 0.337867301, - 0.3997707643, 0.1648201271])
            LELO3[2, :] = np.array([- 0.009587872354, 0.0003712272422, 0.3260097151, 0.8016481698, 0.4720552497, - 0.1400420768, - 0.08542510419, 0.03521962531])
            RELO3 = np.zeros((3, 8))
            RELO3[0, 0:4] = np.array([0.9096849932, 0.3823606566, 0.1509872202, 0.0589610111])
            RELO3[1, 0:6] = np.array([- 0.2904078626, 0.4189992458, 0.4969643833, 0.4907578162, 0.4643627531, 0.1914505327])
            RELO3[2, :] = np.array([0.08183542639, - 0.1587582353, - 0.09124735588, 0.0006042707194, 0.0770293676, 0.520060179, 0.7642591949, 0.3150938119])
            a = F3
            b = LELO3
            c = RELO3
    
    if str(request) == str('PreCondition'):
        if degree == 2:
            LPREMAT2 = np.array([[0.03715799299, 0.3248940464], [1.001445417, 0.0]])
            RPREMAT2 = np.array([[- 0.8008131776, 1.089843048], [2.096292891, 0.0]])
            a = LPREMAT2
            b = np.flipud(RPREMAT2).T
            c = []
        if degree == 3:
            LPREMAT3 = np.array([[- 0.01509646707, - 0.5929309617, 0.1007941548], [0.03068539981, 0.213725675, 0.0], [1.000189315, 0.0, 0.0]])
            RPREMAT3 = np.array([[2.417783369, - 0.4658799095, 1.055782523], [- 6.66336795, 1.737631831, 0], [5.642212259, 0, 0]])
            a = LPREMAT3
            b = np.flipud(RPREMAT3).T
            c = []
    
    if str(request) == str('PostCondition'):
        if degree == 2:
            RPOSTMAT2 = np.array([[0.0, 0.4770325771], [0.9175633147, 0.3505220082]])
            LPOSTMAT2 = np.array([[0.0, 0.9985566693], [3.077926515, - 0.1142044987]])
            a = LPOSTMAT2
            b = np.flipud(np.transpose(RPOSTMAT2))
            c = []
        if degree == 3:
            LPOSTMAT3 = np.array([[0.0, 0.0, 0.999810721], [0.0, 4.678895037, - 0.1435465894], [9.921210229, 27.52403389, - 0.6946792478]])
            RPOSTMAT3 = np.array([[0, 0, 0.1772354449], [0, 0.5754959032, 0.6796520196], [0.9471647601, 0.2539462185, - 0.1059694467]])
            a = np.flipud(LPOSTMAT3)
            b = np.flipud(np.transpose(RPOSTMAT3))
            c = []
    
    return a, b, c
    
    # Copyright (c) 1993. David L. Donoho
    
    #  Part of Wavelab Version 850
    #  Built Tue Jan  3 13:20:40 EST 2006
    #  This is Copyrighted Material
    #  For Copying permissions see COPYING.m
    #  Comments? e-mail wavelab@stat.stanford.edu


# Demo:
if __name__ == '__main__':
    a, b, c = MakeCDJVFilter(request='HighPass', degree=2)
