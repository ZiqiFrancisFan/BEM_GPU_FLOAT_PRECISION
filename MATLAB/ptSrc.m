function [result] = ptSrc(strength,wavNum,x,y)
%UNTITLED5 Summary of this function goes here
%   x is a 3-tuple source location
r = norm(x-y,2);
result = complex(strength*cos(-wavNum*r)/(4*pi*r),strength*sin(-wavNum*r)/(4*pi*r));
end

