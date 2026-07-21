function simpleflowchart()
    % SIMPLEFLOWCHART Generates a simplified staggered flowchart for our image encryption algorithm.
    % This script draws the boxes and arrows in a staircase layout similar to the example.
    
    % Initialize figure
    clf;
    figure('Color', 'w', 'Name', 'Simple Flowchart', 'Position', [100, 100, 750, 650]);
    hold on;
    axis equal;
    axis([0 10 -1.5 8]);
    axis off;
    
    % Draw boxes
    % Column 1 (x=2)
    draw_box('Original Image', 2.0, 7.0, 2.0, 0.7, [1.0, 0.95, 0.8], [0.84, 0.71, 0.34], true);
    draw_box({'SHA-256', 'Key Derivation'}, 2.0, 5.5, 2.0, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'4D-DSCHM', 'Chaos Generator'}, 2.0, 4.0, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    
    % Column 2 (x=6)
    draw_box({'Row & Column', 'Scrambling'}, 6.0, 4.0, 2.0, 0.7, [1, 1, 1], [0, 0, 0], false);
    
    % Column 1.5 (x=4)
    draw_box({'RB-SVRCA', 'Iteration 1'}, 4.0, 2.5, 2.0, 0.7, [1, 1, 1], [0, 0, 0], false);
    
    % Column 2.5 (x=8)
    draw_box({'RB-SVRCA', 'Iteration 2'}, 8.0, 2.5, 2.0, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Bidirectional', 'Feedback Diffusion'}, 8.0, 1.0, 2.0, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box('Cipher Image', 8.0, -0.5, 2.0, 0.7, [0.96, 0.96, 0.96], [0.5, 0.5, 0.5], true);
    
    % Draw Arrows
    % Original Image -> SHA-256
    draw_arrow(2.0, 6.65, 2.0, 5.85, '-');
    % SHA-256 -> 4D-DSCHM
    draw_arrow(2.0, 5.15, 2.0, 4.35, '-');
    % 4D-DSCHM -> Scrambling
    draw_arrow(3.1, 4.0, 5.0, 4.0, '-');
    % Scrambling -> Iteration 1 (bent)
    draw_bent_arrow([6.0, 3.65; 6.0, 3.25; 4.0, 3.25; 4.0, 2.85], '-');
    % Iteration 1 -> Iteration 2
    draw_arrow(5.0, 2.5, 7.0, 2.5, '-');
    % Iteration 2 -> Bidirectional Feedback Diffusion
    draw_arrow(8.0, 2.15, 8.0, 1.35, '-');
    % Bidirectional Feedback Diffusion -> Cipher Image
    draw_arrow(8.0, 0.65, 8.0, -0.15, '-');
    
    hold off;
end

% Draw Rectangle box helper
function draw_box(txt, x, y, w, h, bg_color, border_color, is_bold)
    rect_pos = [x - w/2, y - h/2, w, h];
    rectangle('Position', rect_pos, 'EdgeColor', border_color, 'FaceColor', bg_color, 'LineWidth', 1.5);
    
    if is_bold
        text(x, y, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9.5, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(x, y, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9.5, 'Color', 'k');
    end
end

% Draw Arrow helper
function draw_arrow(x1, y1, x2, y2, line_style)
    plot([x1, x2], [y1, y2], 'Color', 'k', 'LineStyle', line_style, 'LineWidth', 1.5);
    dx = x2 - x1;
    dy = y2 - y1;
    len = sqrt(dx^2 + dy^2);
    if len > 0
        dx = dx / len;
        dy = dy / len;
        arrow_len = 0.22;
        arrow_width = 0.09;
        nx = -dy;
        ny = dx;
        px1 = x2 - arrow_len * dx + arrow_width * nx;
        py1 = y2 - arrow_len * dy + arrow_width * ny;
        px2 = x2 - arrow_len * dx - arrow_width * nx;
        py2 = y2 - arrow_len * dy - arrow_width * ny;
        fill([x2, px1, px2], [y2, py1, py2], 'k', 'EdgeColor', 'k', 'HandleVisibility', 'off');
    end
end

% Draw Bent Arrow helper
function draw_bent_arrow(pts, line_style)
    if nargin < 2, line_style = '-'; end
    for i = 1:size(pts, 1)-1
        x1 = pts(i, 1); y1 = pts(i, 2);
        x2 = pts(i+1, 1); y2 = pts(i+1, 2);
        if i == size(pts, 1)-1
            draw_arrow(x1, y1, x2, y2, line_style);
        else
            plot([x1, x2], [y1, y2], 'Color', 'k', 'LineStyle', line_style, 'LineWidth', 1.5);
        end
    end
end
