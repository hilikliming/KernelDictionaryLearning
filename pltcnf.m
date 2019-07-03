function [ output_args ] = pltcnf( g,ghat,nmlz,prec )
% % Jack Hall
% A function to plot regular and normalized confusion matrices,
% adapted from following matrix intensity visualization script:
% http://stackoverflow.com/questions/3942892/how-do-i-visualize-a-matrix-with-colors-and-values-displayed
%
% Inputs:
%   g - true class labels
% ghat- predicted class labels
% nmlz- boolean flag indicating whether to normalize the matrix by # of samples
% prec- string- if specified, will give desired decimal precision to
% display for normalized matrices (e.g. '%0.2f' or '%0.3f'
%
%
M=length(unique(g));
if nargin <=3
    prec='%0.2f';
end

if nargin <=2
    norm=false;
end
[cm,gm]=confusionmat(g,ghat);
xlbl=cell(0);ylbl=cell(0);
for row=1:size(cm,1)
    if nmlz
        cm(row,:)=cm(row,:)/sum(cm(row,:));
    else
        cm(row,:)=(cm(row,:));
    end
    xlbl{end+1}=num2str(gm(row));
    ylbl{end+1}=num2str(gm(row));
end
cm(isnan(cm))=0;

imagesc(cm);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

if nmlz                         
    textStrings = num2str(cm(:),prec);  %# Create strings from the matrix values
else
    textStrings = num2str(cm(:));
end
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(cm,2));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(cm(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
xlabel('Output Class');ylabel('Target Class');
title('Confusion Matrix');
if M>=10
    cell_guy=strsplit(num2str(0:M-1));
else
    cell_guy=squeeze(char(string(num2cell((0:M-1)))));
end

set(gca,'XTick',1:M,...                         %# Change the axes tick marks
        'XTickLabel',cell_guy,...  %#   and tick labels
        'YTick',1:M,...
        'YTickLabel',cell_guy,...
        'TickLength',[0 0]);

end

