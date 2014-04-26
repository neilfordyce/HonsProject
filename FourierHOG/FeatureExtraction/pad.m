function [paddedData,shift] = pad(data,shape)

if ndims(data) == 3
    paddedData = zeros(shape);
    
    c = shape / 2 + 0.5;
    
    ds = size(data);
    
    hds = ds / 2 - 0.5;
    
    startPoint = round(c - hds);
    
    endPoint = startPoint + ds - 1;
    
    paddedData(startPoint(1):endPoint(1),startPoint(2):endPoint(2),startPoint(3):endPoint(3)) = data;
    
else
    
    paddedData = zeros(shape);
    
    c = shape / 2 + 0.5;
    
    ds = size(data);
    
    hds = ds / 2 - 0.5;
    
    startPoint = round(c - hds);
    
    endPoint = startPoint + ds - 1;
    
    paddedData(startPoint(1):endPoint(1),startPoint(2):endPoint(2)) = data;
    
end

shift = startPoint - 1;