function [a,b,c] = MakeCDJVFilter(request,degree)
% MakeCDJVFilter --  Set up filters for CDJV Wavelet Transform
%  Usage
%    [a,b,c] = MakeCDJVFilter(request,degree)
%  Inputs
%    request  string: 'HighPass', 'LowPass', 'Precondition', 'Postcondition'
%    degree   integer: 2 or 3 (number of vanishing moments)
%  Outputs
%    a,b,c    filter, left edge filter, right edge filter
%             ('HighPass', 'LowPass')
%    a        conditioning matrix ('Precondition', 'Postcondition')
%
%  Description
%    CDJV have developed an algorithm for wavelets on the interval which
%    preserves the orthogonality, vanishing moments, smoothness, and compact
%    support of Daubechies wavelets on the line.
%
%    The algorithm for wavelets on the interval of CDJV involves four objects
%    not present in the usual periodized algorithm: right edge filters, left
%    edge filters, and pre- and post- conditioning operators.
%
%    These objects are supplied by appropriate requests to MakeCDJVFilter.
%
%  See Also
%    IWT_CDJV, FWT_CDJV, CDJVDyadDown
%
%  References
%    Cohen, Daubechies, Jawerth and Vial, 1992.
%
F2 = [  0.482962913145 0.836516303738 0.224143868042 -0.129409522551];
F3 = [  .332670552950 .806891509311 .459877502118 -.135011020010  ...
-.085441273882 .035226291882 ];
if strcmp(request,'HighPass'),
    if degree==2,
         LEHI2 = [ -.7965435153E+00  .5463927105  -.2587922607 0 0; ...
             .01003722199 .1223510414 .2274281035  -.8366029193 .4830129294 ];
         REHI2 = [ -.2575129317 .8014229647 -.5398224908 0 0 ; ...
              .3717189691 -.3639069552  -.7175800176 .4010694996 .2315575756 ];
        a = reverse(MirrorFilt(F2)); b = LEHI2; c = REHI2;
     end
     if degree==3,
       LEHI3 = [ .5837810161 .7936188102 .1609551602 -.05884169984 0 0 0 0 ; ...
                -.3493401755 .2989205708 -.3283012959 -.3322637280 ...
                 .6982497314 -.2878790040 0 0; ...
                 .001015059936 -.00003930151414 -.03451437279 -.08486981368 ... 
                 .1337306925    .4604064313 -.8068932340 .3326712638 ];
        REHI3 = zeros(3,8);
        REHI3(1,1:4) = [   0.07221947896  -0.42656220040000   ...
                           0.80423313630  -0.40747772770000 ];
        REHI3(2,1:6) = [  -0.1535052177   0.5223942253  -0.09819804815 ...  
                          -0.7678795675   0.2985152672   0.1230738394];
        REHI3(3,:)   = [ 0.2294775468   -0.4451794532  -0.2558698634  ... 
                         0.001694456403  0.7598761492   0.1391503023  ...              
                        -0.2725472621   -0.1123675794];
        a = reverse(MirrorFilt(F3)); b = LEHI3; c = REHI3;
     end
end
if strcmp(request,'LowPass'),
    if degree==2,
         LELO2 = [.6033325147 .6908955290  -.3983129985 0 0 ;   ...
          .3751745208e-01 .4573276687 .8500881006 .2238203490 -.1292227411  ] ;
         RELO2 =  [ .8705087515 .4348970037 .2303890399 0 0;
               -.1942333944 .1901514021 .3749553135 .7675566880 .4431490452 ]; 
        a = F2; b = LELO2; c = RELO2;
     end
     if degree==3,
       LELO3 = zeros(3,8);
       LELO3(1,1:4) = [.3888996730 -.08820780195 -.8478413443 .3494874575];
       LELO3(2,1:6) = [ -.6211483347  .5225274354 -.2000079353 ...
                         .3378673010 -.3997707643  .1648201271];
       LELO3(3,:) = [  -.009587872354 .0003712272422 .3260097151  ...
       .8016481698 .4720552497 -.1400420768 -.08542510419 .03521962531 ];
       RELO3 = zeros(3,8);
       RELO3(1,1:4) = [ 0.9096849932 0.3823606566 0.1509872202 0.0589610111];
       RELO3(2,1:6) = [ -0.2904078626 0.4189992458  0.4969643833 ...  
                         0.4907578162  0.4643627531   0.1914505327 ];
       RELO3(3,:)   = [ 0.08183542639  -0.1587582353  -0.09124735588  ... 
          0.00060427071940 0.0770293676   0.520060179   0.7642591949  ...  
          0.3150938119 ];
        a = F3; b = LELO3; c = RELO3;
     end
end
if strcmp(request,'PreCondition'),
    if degree==2,
      LPREMAT2 = [     .03715799299     .3248940464   ; ... 
                      1.001445417       .0000000000  ];
      RPREMAT2 =  [    -.8008131776      1.089843048 ; ...
                       2.096292891      .00000000 ];
       a = LPREMAT2; b = flipud(RPREMAT2)'; c = [];
     end
     if degree==3,
       LPREMAT3 = [    -.01509646707 -.5929309617  .1007941548 ; ...
                        .03068539981  .2137256750  .0000000000 ; ...
                       1.000189315    .0000000000  .0000000000 ];
     RPREMAT3 = [2.417783369        -.4658799095           1.055782523  ;... 
                -6.66336795000000   1.73763183100000                  0 ;...
                 5.64221225900000                  0                  0 ];
        a = fliplr(LPREMAT3); b = flipud(RPREMAT3)'; c = [];
     end
end
if strcmp(request,'PostCondition'),
    if degree==2,
        RPOSTMAT2 = [ .0000000000    .4770325771 ;
                      .9175633147    .3505220082 ];
        LPOSTMAT2 = [ 0.000000000         .9985566693; ...
                      3.077926515        -.1142044987 ];
        a = LPOSTMAT2; b = flipud(RPOSTMAT2'); c = [];
     end
     if degree==3,
       LPOSTMAT3 = [  .0000000000  .000000000  .9998107210 ;...
                      .0000000000 4.678895037 -.1435465894 ;...
                     9.921210229 27.52403389  -.6946792478 ];
       RPOSTMAT3 = [ 0                0   0.1772354449 ;...
                     0     0.5754959032   0.6796520196 ;...
               0.9471647601   0.2539462185  -0.1059694467 ];
        a = flipud(LPOSTMAT3); b = flipud(RPOSTMAT3'); c = [];
     end
end

%
% Copyright (c) 1993. David L. Donoho
%    
    
    

    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:40 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
