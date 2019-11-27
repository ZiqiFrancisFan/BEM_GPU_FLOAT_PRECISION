% function that creates pink noise

function pink = makePinkNoise(dur, srate)

% Code from

% https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html

npts = dur * srate;  % number of samples to synthesize

B = [0.049922035 -0.095993537 0.050612699 -0.004408786];

A = [1 -2.494956002   2.017265875  -0.522189400];

nT60 = round(log(1000)/(1-max(abs(roots(A))))); % T60 estimation.

v = randn(1,npts+nT60); % Gaussian white noise: N(0,1)

x = filter(B,A,v);    % Apply 1/F roll-off to PSD

pink = x(nT60+1:end);    % Skip transient response