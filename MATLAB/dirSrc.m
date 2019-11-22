function result = dirSrc(strength,wavNum,dir,pt)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
eps = 0.001;
if(length(dir)~=3 || length(pt)~=3 || abs(dot(dir,dir)-1)>eps)
    fprintf('Incorrect input format.\n');
    return;
end
theta = -wavNum*dot(dir,pt);
result = complex(strength*cos(theta),strength*sin(theta));
end

