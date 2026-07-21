function graphicalabstract()
    % GRAPHICALABSTRACT Generates the IEEE-style flowchart for the image cryptosystem.
    % This script draws the boxes, solid flow arrows, dashed keystream arrows,
    % loads plain/cipher thumbnails, and displays a legend box.
    
    % Initialize figure
    clf;
    figure('Color', 'w', 'Name', 'Graphical Abstract', 'Position', [100, 100, 800, 800]);
    hold on;
    
    % Coordinates setup
    col1_x = 2.6;
    col2_x = 6.8;
    
    % 1. Load and plot actual project images FIRST (as background)
    plain_img_path = 'Images/lena_color_512.tif';
    if ~exist(plain_img_path, 'file')
        plain_img_path = 'Images/lena_gray_512.tif';
    end
    
    cipher_img_path = 'results/encrypted/enc_lenna.png';
    if ~exist(cipher_img_path, 'file')
        cipher_img_path = 'ciphertext.png';
    end
    
    % Draw Plain Image Thumbnail
    if exist(plain_img_path, 'file')
        img_plain = imread(plain_img_path);
        if size(img_plain, 3) == 1
            img_plain = cat(3, img_plain, img_plain, img_plain);
        end
        imagesc([0.4, 1.2], [10.8, 11.6], img_plain);
        rectangle('Position', [0.4, 10.8, 0.8, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Draw Cipher Image Thumbnail
    if exist(cipher_img_path, 'file')
        img_cipher = imread(cipher_img_path);
        if size(img_cipher, 3) == 1
            img_cipher = cat(3, img_cipher, img_cipher, img_cipher);
        end
        imagesc([0.4, 1.2], [1.0, 1.8], img_cipher);
        rectangle('Position', [0.4, 1.0, 0.8, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
    end
    
    % Enforce axes configuration and limits
    axis equal;
    axis([0 10.5 0 12.5]);
    axis off;
    
    % 2. Draw boxes
    draw_box({'Plain Image', '(M x N)'}, col1_x, 11.2, 2.2, 0.7, [1.0, 0.95, 0.8], [0.84, 0.71, 0.34], true);
    draw_box({'Padding &', 'Block Setup'}, col1_x, 9.9, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Row & Column', 'Scrambling (Confusion)'}, col1_x, 8.4, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'RB-SVRCA', 'CA Iteration 1'}, col1_x, 7.0, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'RB-SVRCA', 'CA Iteration 2'}, col1_x, 5.6, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Bidirectional Feedback', 'Diffusion (Forward)'}, col1_x, 4.2, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Bidirectional Feedback', 'Diffusion (Backward)'}, col1_x, 2.8, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Cipher Image', '(M x N)'}, col1_x, 1.4, 2.2, 0.7, [0.85, 0.91, 0.99], [0.42, 0.56, 0.75], true);
    
    draw_box({'256-bit Secret Key', '(K)'}, col2_x, 11.2, 2.2, 0.7, [0.96, 0.96, 0.96], [0.7, 0.7, 0.7], true);
    draw_box({'SHA-256 Key Derivation', '(Digest)'}, col2_x, 9.9, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'Initial States', '(x0, y0, z0, w0)'}, col2_x, 8.6, 2.2, 0.7, [1, 1, 1], [0, 0, 0], false);
    draw_box({'4D-DSCHM', 'Hyperchaotic Map'}, col2_x, 7.2, 2.2, 0.9, [1, 1, 1], [1, 0.6, 0], false);
    
    % Draw Solid Connections (Main Flow)
    draw_bent_arrow([col1_x, 10.85; col1_x, 10.25], '-');
    draw_bent_arrow([col1_x, 9.55; col1_x, 8.75], '-');
    draw_bent_arrow([col1_x, 8.05; col1_x, 7.35], '-');
    draw_bent_arrow([col1_x, 6.65; col1_x, 5.95], '-');
    draw_bent_arrow([col1_x, 5.25; col1_x, 4.55], '-');
    draw_bent_arrow([col1_x, 3.85; col1_x, 3.15], '-');
    draw_bent_arrow([col1_x, 2.45; col1_x, 1.75], '-');
    
    draw_bent_arrow([col2_x, 10.85; col2_x, 10.25], '-');
    draw_bent_arrow([col2_x, 9.55; col2_x, 8.95], '-');
    draw_bent_arrow([col2_x, 8.25; col2_x, 7.65], '-');
    
    % Draw Dashed Connections (Keystreams)
    % 1. 4D-DSCHM -> Scrambling
    draw_bent_arrow([col2_x - 1.1, 7.4; 4.8, 7.4; 4.8, 8.4; col1_x + 1.1, 8.4], '--', ...
                    'xs, ys', 4.6, 8.55, 'center');
    
    % 2. 4D-DSCHM -> CA Iteration 1 (zs)
    draw_bent_arrow([col2_x - 1.1, 7.2; 4.5, 7.2; 4.5, 7.0; col1_x + 1.1, 7.0], '--', ...
                    'zs', 4.3, 7.15, 'center');
    
    % 3. 4D-DSCHM -> CA Iteration 2 (ws)
    draw_bent_arrow([col2_x - 1.1, 7.0; 4.2, 7.0; 4.2, 5.6; col1_x + 1.1, 5.6], '--', ...
                    'ws', 4.0, 5.75, 'center');
                    
    % 4. SHA-256 Digest -> Feedback Diffusion (IV parameters)
    draw_bent_arrow([col2_x + 1.1, 9.9; 8.2, 9.9; 8.2, 4.2; col1_x + 1.1, 4.2], '--', ...
                    'IV1, IV2', 4.8, 4.35, 'center');
                    
    % Connect Image thumbnails to processing path
    if exist(plain_img_path, 'file')
        draw_bent_arrow([1.2, 11.2; col1_x - 1.1, 11.2]);
    end
    if exist(cipher_img_path, 'file')
        draw_bent_arrow([col1_x - 1.1, 1.4; 1.2, 1.4]);
    end

    % Draw Legend Box (bottom right corner)
    legend_x = 8.8;
    legend_y = 2.2;
    legend_w = 3.0;
    legend_h = 2.8;
    rectangle('Position', [legend_x - legend_w/2, legend_y - legend_h/2, legend_w, legend_h], ...
              'EdgeColor', [0.3, 0.3, 0.3], 'FaceColor', [0.98, 0.98, 0.98], 'LineWidth', 1.5);
          
    text(legend_x, legend_y + 1.1, 'LEGEND & TERMINOLOGY', 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
    plot([legend_x - legend_w/2 + 0.2, legend_x + legend_w/2 - 0.2], [legend_y + 0.9, legend_y + 0.9], ...
         'Color', [0.6, 0.6, 0.6], 'LineWidth', 1);
     
    legend_text = { ...
        '\textbf{xs, ys} : Row \& column scrambling maps', ...
        '\textbf{zs}         : Space-varying CA rules grid', ...
        '\textbf{ws}        : Cellular Automata key grid', ...
        '\textbf{IV1, IV2} : Initial diffusion vectors', ...
        '\textbf{CA}       : Cellular Automata updates', ...
        '\textbf{DSCHM}  : Sine-Cosine Hyperchaotic Map', ...
        '\textbf{RB-SVRCA}: Red-Black Space-Varying', ...
        '                Reversible CA' ...
    };
    text(legend_x - 1.35, legend_y - 0.2, legend_text, 'HorizontalAlignment', 'left', ...
         'VerticalAlignment', 'middle', 'FontSize', 8.5, 'Interpreter', 'latex');
    
    hold off;
end

% Draw Rectangle box helper
function draw_box(txt, x, y, w, h, bg_color, border_color, is_bold)
    rect_pos = [x - w/2, y - h/2, w, h];
    rectangle('Position', rect_pos, 'EdgeColor', border_color, 'FaceColor', bg_color, 'LineWidth', 1.5);
    
    if is_bold
        text(x, y, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(x, y, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9, 'Color', 'k');
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
function draw_bent_arrow(pts, line_style, label_text, lx, ly, align_text)
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
    if nargin >= 5 && ~isempty(label_text)
        text(lx, ly, label_text, 'HorizontalAlignment', align_text, 'VerticalAlignment', 'middle', ...
             'FontSize', 8.5, 'FontAngle', 'italic', 'BackgroundColor', 'w');
    end
end
