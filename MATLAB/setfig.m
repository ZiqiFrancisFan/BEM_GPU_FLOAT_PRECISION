function setfig(fig, axes, h, pos)
% make figure look "good"

% Created by Hannes Gamper

if nargin<4
    pos = [];
end

set(axes, 'FontName', 'Times New Roman', 'FontSize', 8);

xgrid = get(axes, 'xgrid');
ygrid = get(axes, 'ygrid');
zgrid = get(axes, 'zgrid');

if isfield(h, 'interpreter')
    interpreter = h.interpreter;
else
    interpreter = 'tex';
end

if isfield(h, 'unit')
    unit = h.unit;
else
    unit = 'inches';
end

if isfield(h, 'title')
    set(h.title, 'FontName', 'Times New Roman', 'FontSize', 9, ...
        'Color', [.3 .3 .3], 'Interpreter', interpreter);
end
if isfield(h, 'xlabel')
    set(h.xlabel, 'FontName', 'Times New Roman', 'FontSize', 9, ...
        'Interpreter', interpreter);
end
if isfield(h, 'ylabel')
    set(h.ylabel, 'FontName', 'Times New Roman', 'FontSize', 9, ...
        'Interpreter', interpreter);
end
if isfield(h, 'zlabel')
    set(h.zlabel, 'FontName', 'Times New Roman', 'FontSize', 9, ...
        'Interpreter', interpreter);
end
if isfield(h, 'legend')
    set(h.legend, 'FontSize', 8, 'LineWidth', 0.5, ...
        'Interpreter', interpreter);
end
if isfield(h, 'text')
    set(h.text, 'FontName', 'Times New Roman', 'FontSize', 8, ...
        'Color', [.3 .3 .3], 'Interpreter', interpreter);
    set(h.text,'HorizontalAlignment','center');
end

if isfield(h,'smalltext')
    set(h.smalltext, 'FontName', 'Times New Roman', 'FontSize', 6, ...
        'Color', [.3 .3 .3], 'Interpreter', interpreter);
    %fprintf('upper text used\n');
    set(h.smalltext,'HorizontalAlignment','center');
end

if isfield(h,'colorbar')
    set(h.colorbar,'FontName','Times New Roman','FontSize',6,'YColor',[.3 .3 .3]);
    if isfield(h,'colorbarwidth') && isfield(h,'cb2imgspace')
        cp = get(h.colorbar, 'Position'); % get the position of the colorbar
        ap = get(gca, 'Position'); % get the position of the current axis
        fp = get(gcf, 'Position'); % get the position of the current figure
        set(h.colorbar, 'Position',...
            [ap(1)+ap(3)+h.cb2imgspace,cp(2),h.colorbarwidth/fp(3),cp(4)],'Units','normalized');
        set(gca, 'Position', ap);
    end
    if isfield(h,'cblower') && isfield(h,'cbupper')
        caxis(gca,[h.cblower,h.cbupper]);
    end
    if isfield(h,'cbtitle')
        set(h.cbtitle,'Units','normalized');
        cbttpos = get(h.cbtitle,'Position');
        %cbpos = get(h.colorbar,'Position');
        %cbttpos(2) = 0.99*cbttpos(2);
        cbttpos(2) = 1.0;
        set(h.cbtitle,'Position',cbttpos);
        set(h.cbtitle,'FontName','Times New Roman','FontSize',5);
    end
end

if nargin < 4 || isempty(pos)
    pos = [2 2 10 7];
end
if isfield(h, 'pos')
    pos = h.pos;
end
if isfield(h, 'Position')
    pos = h.Position;
end

% fprintf('Unit: %s, Position: [%f,%f,%f,%f]\n',fig.Units,pos(1),pos(2),pos(3),pos(4));

if isfield(h, 'xgrid')
    xgrid = h.xgrid;
end

if isfield(h, 'ygrid')
    ygrid = h.ygrid;
end

if isfield(h, 'zgrid')
    zgrid = h.zgrid;
end

set(axes, ...
    'Box'         , 'off'     , ...
    'TickDir'     , 'out'     , ...
    'TickLength'  , [.02 .02] , ...
    'XGrid'       , xgrid     , ...
    'YGrid'       , ygrid     , ...
    'ZGrid'       , zgrid     , ...
    'XColor'      , [.3 .3 .3], ...
    'YColor'      , [.3 .3 .3], ...
    'ZColor'      , [.3 .3 .3], ...
    'LineWidth'   , 1);
set(fig, 'Unit', unit);
set(fig, 'Position', pos);
set(fig, 'PaperPositionMode', 'auto');
%set(fig, 'Renderer', 'Painters');
if isfield(h, 'yMinorGrid')
    set(axes, 'YMinorGrid', h.yMinorGrid);
    set(axes, 'YMinorTick', h.yMinorGrid);
end

end
