function showboxes(im, boxes, det_count, out)

% showboxes(im, boxes, out)
% Draw bounding boxes on top of image.
% If out is given, a pdf of the image is generated (requires export_fig).

if nargin > 3
  % different settings for producing pdfs
  print = true;
  cwidth = 0.5;
else
  print = false;
  cwidth = 0.5;
end
hh=figure(99);
% clf;
imshow((im)); 
colormap gray;
truesize;
axis equal;
axis off;
set(gcf, 'Color', 'white');

if ~isempty(boxes)
  numfilters = floor(size(boxes, 2)/4);
%{
  if print
    % if printing, increase the contrast around the boxes
    % by printing a white box under each color box
    for i = 1:numfilters
      x1 = boxes(:,1+(i-1)*4);
      y1 = boxes(:,2+(i-1)*4);
      x2 = boxes(:,3+(i-1)*4);
      y2 = boxes(:,4+(i-1)*4);
      % remove unused filters
      del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
      x1(del) = [];
      x2(del) = [];
      y1(del) = [];
      y2(del) = [];
      if i == 1
        w = 1;
      else
        w = 1;
      end
      line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'w', 'linewidth', wwidth);
    end
  end
  %}
  % draw the boxes with the detection window on top (reverse order)
  for i = 1:det_count
    x1 = boxes(:,1+(i-1)*4);
    y1 = boxes(:,2+(i-1)*4);
    x2 = boxes(:,3+(i-1)*4);
    y2 = boxes(:,4+(i-1)*4);
    % remove unused filters
    del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
    x1(del) = [];
    x2(del) = [];
    y1(del) = [];
    y2(del) = [];
    if i == 1
      c = [160/255 0 0];
      s = '-';
    else
      c = 'b';
      s = '-';
    end

    for j = 1:numel(x1)
         v = max(min(boxes(j,5), 1),0);
        line([x1(j) x1(j) x2(j) x2(j) x1(j)]', [y1(j) y2(j) y2(j) y1(j) y1(j)]', 'color', [1 - v,v,0], 'linewidth', cwidth, 'linestyle', s);
    end
    
  end
end

% save to pdf
if print
  % requires export_fig from http://www.mathworks.com/matlabcentral/fileexchange/23629-exportfig
%     saveas(hh,num2str(out), 'png')
  export_fig((out), '-png')
end
