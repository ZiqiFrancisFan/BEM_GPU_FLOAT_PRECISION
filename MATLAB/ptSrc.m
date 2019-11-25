function [result] = ptSrc(strength,wavNum,x,y)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
r = norm(x-y,2);
result = complex(strength*cos(-wavNum*r)/(4*pi*r),strength*sin(-wavNum*r)/(4*pi*r));
end

