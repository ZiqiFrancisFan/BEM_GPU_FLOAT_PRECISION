function y = mpSrc(strength,wavNum,x,y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
rho_0 = 1.2041;
c_0 = 343.21;
y = complex(0,strength*rho_0*c_0*wavNum)*ptSrc(1.0,wavNum,x,y);
end

