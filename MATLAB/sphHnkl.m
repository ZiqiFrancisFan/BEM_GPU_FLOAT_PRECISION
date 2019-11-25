function [y] = sphHnkl(n,x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
y = sqrt(pi/(2*x))*besselh(n+0.5,x);
end

