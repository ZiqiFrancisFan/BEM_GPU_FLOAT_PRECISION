function [output] = ConvertBoundary2SolidObject(input)
%ConvertBoundary2SolidObject converts a boundary map to a map with solid
%objects. input: an input binary map with boundaries denoted by 1 and air
%denoted by 0; output: a binary map with objects denoted by 1 and air
%denoted by 0. We assume that the map starts with air and ends with air in
%both horizontal and vertical directions

hDim = size(input,2); % iterated by the variable x
vDim = size(input,1); % iterated by the variable y

leftBoundaryStart = 0;
leftBoundaryEnd = 0;

rightBoundaryStart = 0;
rightBoundaryEnd = 0;

y = 1;

output = input;

while(y <= size(input,1))
    x = 1;
    leftBoundaryStart = 0;
    leftBoundaryEnd = 0;

    rightBoundaryStart = 0;
    rightBoundaryEnd = 0;
    while(x < size(input,2) - 1)
        
        if(leftBoundaryStart==0 && input(y,x)==0 && input(y,x+1)==1)
            leftBoundaryStart = x + 1;
            xTemp = leftBoundaryStart;
            while(leftBoundaryEnd==0)
                xTemp = xTemp + 1;
                if(input(y,xTemp) == 0)
                    leftBoundaryEnd = xTemp - 1; % found the end of the left boundary of the geometry
                    x = leftBoundaryEnd + 1;
                end
            end
        end
        
        if(leftBoundaryEnd~=0 && rightBoundaryStart==0 && input(y,x)==1)
            rightBoundaryStart = x;
            xTemp = rightBoundaryStart;
            while(rightBoundaryEnd == 0)
                xTemp = xTemp + 1;
                if(input(y,xTemp)==0)
                    rightBoundaryEnd = xTemp - 1;
                    x = rightBoundaryEnd + 1;
                end
            end
        end
        
        if(leftBoundaryEnd ~= 0 && rightBoundaryStart ~= 0)
            for xTemp = leftBoundaryEnd + 1 : rightBoundaryStart - 1
                output(y,xTemp) = 1;
            end
            leftBoundaryStart = 0;
            leftBoundaryEnd = 0;
            rightBoundaryStart = 0;
            rightBoundaryEnd = 0;
        end
        
        x = x + 1;
    end
    y = y + 1;
end





end

