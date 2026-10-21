% Phase E4 copy of fig3_coded.m (2026-10-20): reads figdata/fig3_e4_data.json (verifiers/analysis/make_fig_data.py), writes figs/fig3_coded_e4.*.
% ISBI 2027 Figure 3 (replaces Table 1, 2026-10-20, Omer: "this table is completely unreadable"): how much extra SNR
% simulated median physics needs, as a dot plot (coded-pulse rows removed from the ISBI paper by Omer; kept in fig3_coded_withcodes.m).
% 2026-10-21, Omer: the "goal in ... h" labels inside the dot plot were confusing; training to the goal is now its own small panel.
% A: contrastive verifier evidence per 20-s window after 15 h of training, one dot per participant; dotted line = the 1-bit goal.
% B: training needed to reach 1 bit, same rows; < = reached at the first amount tried (0.8 h) or earlier; "not reached" = not within 15 h.
% Run from the repository root: matlab -batch "run('figures/fig3_coded_e4.m')"   (add -licmode onlinelicensing for an online licence)
P = [fileparts(mfilename('fullpath')) filesep];   % the figures/ directory of this repository
if ~exist([P 'figs'], 'dir'), mkdir([P 'figs']); end
D = jsondecode(fileread([P 'figdata/fig3_e4_data.json']));   % K = 2 (Phase E4)
FN = 'Liberation Sans'; FS = 7;
set(groot, 'defaultAxesFontName', FN, 'defaultTextFontName', FN, 'defaultAxesFontSize', FS, 'defaultTextFontSize', FS, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off', 'defaultAxesLineWidth', 0.6, 'defaultAxesTickLength', [0.025 0.025]);
INK = [0.15 0.15 0.15]; GREY = [0.55 0.55 0.55];
R = {'usL50', 'median physics', INK;
     'usL50u6', '+6 dB signal', INK;
     'usL50u12', '+12 dB signal', INK;
     'usL50u18', '+18 dB signal', INK};
Y = [3.6 2.5 1.5 0.5]; off = [-0.17 0 0.17]; YL = [-0.1 4.3];   % optimistic row removed (duplicates Fig. 2; Omer 2026-10-21)
fig = figure('Units', 'centimeters', 'Position', [1 1 8.6 2.73], 'Color', 'w');
hx = axes(fig, 'Units', 'centimeters', 'Position', [0 0 8.6 2.73], 'XLim', [0 8.6], 'YLim', [0 2.73], 'Visible', 'off'); hold(hx, 'on');
text(hx, 0.45, 2.54, 'Simulated ultrasound, one probe, median physics', 'FontSize', FS + 0.5, 'VerticalAlignment', 'middle');   % Omer 2026-10-21: say which noise

% A: evidence after 15 h
ax = axes(fig, 'Units', 'centimeters', 'Position', [2.4 0.8 3.55 1.30]); hold(ax, 'on');
plot(ax, [1 1], [-0.1 4.2], ':', 'Color', INK, 'LineWidth', 0.8);
text(ax, 1.06, 4.2, 'goal, 1 bit', 'FontSize', FS - 0.5, 'Color', INK, 'VerticalAlignment', 'bottom');
for i = 1:size(R, 1)
    b = D.(R{i, 1}).bits;
    for s = 1:3, plot(ax, b(s), Y(i) + off(s), 'o', 'MarkerSize', 3.6, 'MarkerFaceColor', R{i, 3}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3); end
    text(ax, -0.12, Y(i), R{i, 2}, 'HorizontalAlignment', 'right', 'Color', R{i, 3}, 'FontSize', FS);
end
set(ax, 'XLim', [0 4.5], 'YLim', YL, 'YColor', 'none', 'XTick', 0:1:4);
xlabel(ax, 'Bits after 15 h of training');
letter(ax, 'A', -0.66);

% B: training needed to reach the goal
bx = axes(fig, 'Units', 'centimeters', 'Position', [6.55 0.8 1.85 1.30]); hold(bx, 'on');
for i = 1:size(R, 1)
    [v, k] = hours3(D.(R{i, 1}).hours);
    if all(k == "none"), text(bx, 0.08, Y(i), 'not reached', 'Color', GREY, 'FontSize', FS - 0.5, 'VerticalAlignment', 'middle'); continue; end
    for s = 1:3
        if k(s) == "point", plot(bx, v(s), Y(i) + off(s), 'o', 'MarkerSize', 3.6, 'MarkerFaceColor', R{i, 3}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3);
        elseif k(s) == "below", plot(bx, v(s), Y(i) + off(s), '<', 'MarkerSize', 3.2, 'MarkerFaceColor', R{i, 3}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3); end
    end
end
set(bx, 'XLim', [0 3], 'YLim', YL, 'YColor', 'none', 'XTick', 0:1:3);
xlabel(bx, 'Hours to the goal');
letter(bx, 'B', -0.12);

exportgraphics(fig, [P 'figs/fig3_coded_e4.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, [P 'figs/fig3_coded_e4.png'], 'Resolution', 300, 'BackgroundColor', 'white'); close(fig);
disp('fig3 done');

function [v, k] = hours3(H)
% per participant: hours to 1 bit and its kind ("point"; "below" = at or before the first amount tried; "none" = not within 15 h)
    if iscell(H), n = numel(H); else, n = size(H, 1); end              % jsondecode: cell if any bound is '+inf', else an n x 2 matrix
    v = nan(1, n); k = repmat("none", 1, n);
    for j = 1:n
        if iscell(H), h = H{j}; else, h = H(j, :); end
        if iscell(h), h = cellfun(@(x) ifelse_num(x), h); end
        if numel(h) == 2 && all(isfinite(h)), v(j) = h(2); if h(1) == 0, k(j) = "below"; else, k(j) = "point"; end, end
    end
end
function x = ifelse_num(x)
    if ischar(x) || isstring(x), x = inf; elseif isempty(x), x = nan; end
end
function letter(ax, L, xo)
    text(ax, xo, 1.04, L, 'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'normal', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
end
