% lyapunov.m - Chaos Analysis of 4D Discrete Sine-Cosine Hyperchaotic Map (4D-DSCHM)
% Computes and plots the Lyapunov Exponents spectrum and Bifurcation Diagram.

clear; clc; close all;

%% 1. Parameters Setup
b = 0.354; c = 1.287; d = 0.219; e = 1.543; f = 0.412; g = 1.109; h = 0.331;
a_start = 0.5;
a_end = 3.0;

%% 2. Run Lyapunov Exponents Spectrum Calculation
fprintf('Calculating Lyapunov Exponents Spectrum...\n');
le_steps = 150;
a_vals_le = linspace(a_start, a_end, le_steps);
iterations_le = 1500;
le_results = zeros(le_steps, 4);

for idx = 1:le_steps
    a = a_vals_le(idx);
    
    % Initial states
    x = 0.1; y = 0.2; z = 0.3; w = 0.4;
    Q = eye(4);
    sums = zeros(4, 1);
    
    % Transient warm-up
    for n = 1:500
        x = mod(sin(a * y) + b * cos(x) + w, 1.0);
        y = mod(sin(c * x) + d * cos(y), 1.0);
        z = mod(sin(e * z) + f * cos(w), 1.0);
        w = mod(sin(g * w) + h * cos(z), 1.0);
    end
    
    % Accumulation phase with QR decomposition
    for n = 1:iterations_le
        x = mod(sin(a * y) + b * cos(x) + w, 1.0);
        y = mod(sin(c * x) + d * cos(y), 1.0);
        z = mod(sin(e * z) + f * cos(w), 1.0);
        w = mod(sin(g * w) + h * cos(z), 1.0);
        
        J = get_jacobian(x, y, z, w, a, b, c, d, e, f, g, h);
        [Q, R] = qr(J * Q);
        
        diag_val = abs(diag(R));
        diag_val(diag_val < 1e-12) = 1e-12;
        sums = sums + log(diag_val);
    end
    
    le_results(idx, :) = (sums / iterations_le)';
end

%% 3. Run Bifurcation Diagram Calculation
fprintf('Calculating Bifurcation Diagram...\n');
bif_steps = 450;
a_vals_bif = linspace(a_start, a_end, bif_steps);
transient_bif = 800;
keep_bif = 150;

% Prepare arrays for fast plotting
a_out = zeros(bif_steps * keep_bif, 1);
x_out = zeros(bif_steps * keep_bif, 1);

counter = 1;
for idx = 1:bif_steps
    a = a_vals_bif(idx);
    x = 0.1; y = 0.2; z = 0.3; w = 0.4;
    
    % Transient discard
    for n = 1:transient_bif
        x = mod(sin(a * y) + b * cos(x) + w, 1.0);
        y = mod(sin(c * x) + d * cos(y), 1.0);
        z = mod(sin(e * z) + f * cos(w), 1.0);
        w = mod(sin(g * w) + h * cos(z), 1.0);
    end
    
    % Collect points
    for n = 1:keep_bif
        x = mod(sin(a * y) + b * cos(x) + w, 1.0);
        y = mod(sin(c * x) + d * cos(y), 1.0);
        z = mod(sin(e * z) + f * cos(w), 1.0);
        w = mod(sin(g * w) + h * cos(z), 1.0);
        
        a_out(counter) = a;
        x_out(counter) = x;
        counter = counter + 1;
    end
end

%% 4. Plotting Results
figure('Name', 'Chaos Analysis of 4D-DSCHM', 'Position', [100, 100, 1000, 800]);

% Subplot 1: Lyapunov Exponents
subplot(2, 1, 1);
plot(a_vals_le, le_results(:, 1), 'r-', 'LineWidth', 1.5, 'DisplayName', '\lambda_1 (LE1)');
hold on;
plot(a_vals_le, le_results(:, 2), 'b-', 'LineWidth', 1.5, 'DisplayName', '\lambda_2 (LE2)');
plot(a_vals_le, le_results(:, 3), 'g-', 'LineWidth', 1.5, 'DisplayName', '\lambda_3 (LE3)');
plot(a_vals_le, le_results(:, 4), 'm-', 'LineWidth', 1.5, 'DisplayName', '\lambda_4 (LE4)');
yline(0, 'k--', 'LineWidth', 1.2);
xlabel('Control Parameter \alpha');
ylabel('Lyapunov Exponents');
title('Lyapunov Exponents Spectrum of the 4D-DSCHM');
legend('Location', 'northeast');
grid on;
set(gca, 'FontSize', 11);

% Subplot 2: Bifurcation Diagram
subplot(2, 1, 2);
scatter(a_out, x_out, 1.5, [0, 0, 0.5], 'filled', 'MarkerFaceAlpha', 0.4);
xlabel('Control Parameter \alpha');
ylabel('State Variable x');
title('Bifurcation Diagram of the 4D-DSCHM');
xlim([a_start, a_end]);
ylim([0, 1.0]);
grid on;
set(gca, 'FontSize', 11);

sgtitle('4D Sine-Cosine Hyperchaotic Map Chaos Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Helper Jacobian Function
function J = get_jacobian(x, y, z, w, a, b, c, d, e, f, g, h)
    J = zeros(4, 4);
    J(1, 1) = -b * sin(x);
    J(1, 2) = a * cos(a * y);
    J(1, 3) = 0.0;
    J(1, 4) = 1.0;
    
    J(2, 1) = c * cos(c * x);
    J(2, 2) = -d * sin(y);
    J(2, 3) = 0.0;
    J(2, 4) = 0.0;
    
    J(3, 1) = 0.0;
    J(3, 2) = 0.0;
    J(3, 3) = e * cos(e * z);
    J(3, 4) = -f * sin(w);
    
    J(4, 1) = 0.0;
    J(4, 2) = 0.0;
    J(4, 3) = -h * sin(z);
    J(4, 4) = g * cos(g * w);
end
