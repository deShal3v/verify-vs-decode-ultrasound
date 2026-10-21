% Phase E4 copy of fig2_learning.m (2026-10-20): reads figdata/fig2_data_e4.json (verifiers/analysis/make_fig_data.py), writes figs/fig2_learning_e4.*.
% ISBI 2027 draft, Figure 2: evidence against training data for every condition (none, -6, -12, -18 dB uniform noise; simulated
% ultrasound with one and two probes, optimistic physics), training needed for the goal, and noise survived with the same
% training. Style: plain letters outside the corner, short centred
% titles, decoder red, encode-then-compare blue, contrastive green, mean +- s.e.m. over the three participants).
% Data: the merged grid (11 training amounts). MATLAB only draws.
% Run from the repository root: matlab -batch "run('figures/fig2_learning_e4.m')"   (add -licmode onlinelicensing for an online licence)
P = [fileparts(mfilename('fullpath')) filesep];   % the figures/ directory of this repository
if ~exist([P 'figs'], 'dir'), mkdir([P 'figs']); end
D = jsondecode(fileread([P 'figdata/fig2_data_e4.json']));   % K = 2 grid (Phase E4) + decoder at x63 (E4b)
FN = 'Liberation Sans'; FS = 7;
set(groot, 'defaultAxesFontName', FN, 'defaultTextFontName', FN, 'defaultAxesFontSize', FS, 'defaultTextFontSize', FS, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off', 'defaultAxesLineWidth', 0.6, 'defaultAxesTickLength', [0.02 0.02]);
RED = [0.84 0.15 0.16]; NAVY = [0.12 0.25 0.62]; GREEN = [0.17 0.63 0.17]; INK = [0.25 0.25 0.25]; GREY = [0.45 0.45 0.45];
SUBJ = {'S1', 'S2', 'S3'}; bars = cellfun(@(s) D.(s).bar, SUBJ);
ser = {'ver_ETC', NAVY, 'Verifier, encode-then-compare'; 'ver_C', GREEN, 'Verifier, contrastive'; 'dec_empBig', RED, 'Decoder (Tang et al., 2023)'};
fig = figure('Units', 'centimeters', 'Position', [1 1 17.8 8.75], 'Color', 'w');

% ---------------- legend strip (two rows: readouts, goals)
lx = axes(fig, 'Units', 'centimeters', 'Position', [0.3 7.92 17.3 0.75], 'XLim', [0 17.3], 'YLim', [0 2], 'Visible', 'off'); hold(lx, 'on');
xs = [2.2 7.05 10.6]; yr = 1.5;
for k = 1:3
    plot(lx, xs(k) + [0 0.55], [yr yr], '-', 'Color', ser{k, 2}, 'LineWidth', 1.4);
    text(lx, xs(k) + 0.7, yr, ser{k, 3}, 'FontSize', FS + 0.5, 'VerticalAlignment', 'middle');
end
xg = [5.2 9.6]; yg = 0.45;
plot(lx, xg(1) + [0 0.55], [yg yg], ':', 'Color', INK, 'LineWidth', 1.0);
text(lx, xg(1) + 0.7, yg, 'Verifiers'' goal (1 bit)', 'FontSize', FS + 0.5, 'VerticalAlignment', 'middle');
patch(lx, xg(2) + [0 0.55 0.55 0], yg + [-0.3 -0.3 0.3 0.3], RED, 'FaceAlpha', 0.14, 'EdgeColor', 'none');
text(lx, xg(2) + 0.7, yg, sprintf('Decoder''s goal (published decoder at 15 h, %.1f-%.1f bits)', min(bars), max(bars)), 'FontSize', FS + 0.5, 'VerticalAlignment', 'middle');

% ---------------- learning curves (one y scale everywhere; the decoder's goal is the strip in every panel, Omer 2026-10-20)
CND = {'native', 'm6c', 'm12c', 'm18c', 'usL90', 'usB90'};
TTL = {'No added noise', 'Uniform noise, -6 dB', 'Uniform noise, -12 dB', 'Uniform noise, -18 dB', 'One probe, left temple', 'Two probes, both temples'};
POS = [1.25 5.2; 5.45 5.2; 9.65 5.2; 13.85 5.2; 1.25 1.15; 5.2 1.15]; WW = [3.55 3.55 3.55 3.55 3.3 3.3]; Hh = 2.12;
YM = [8 8 8 8 8 8]; LET = {'A', 'B', 'C', 'D', 'E', 'F'};
for c = 1:6
    ax = axes(fig, 'Units', 'centimeters', 'Position', [POS(c, :) WW(c) Hh]); hold(ax, 'on');
    if YM(c) >= 6
        fill(ax, [0 16 16 0], [min(bars) min(bars) max(bars) max(bars)], RED, 'FaceAlpha', 0.14, 'EdgeColor', 'none');
    elseif c ~= 4
        text(ax, 15.6, YM(c) - 0.05, sprintf('decoder''s goal %.1f-%.1f bits \\uparrow', min(bars), max(bars)), 'Color', RED, ...
             'FontSize', FS, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
    end
    plot(ax, [0 16], [1 1], ':', 'Color', INK, 'LineWidth', 1.0);
    for k = 1:3
        Y = []; X = []; PS = {};
        for s = 1:3
            Pp = D.(SUBJ{s}).(sprintf('cur_%s_%s', CND{c}, ser{k, 1})); if iscell(Pp), Pp = cell2mat(Pp); end
            if isempty(Pp), continue; end
            PS{end + 1} = sortrows(Pp, 1); %#ok<AGROW>
        end
        if isempty(PS), continue; end
        X = PS{1}(:, 1); for s = 2:numel(PS), X = intersect(X, PS{s}(:, 1)); end   % training amounts every participant has
        for s = 1:numel(PS), [~, ix] = ismember(X, PS{s}(:, 1)); Y = [Y, PS{s}(ix, 2)]; end %#ok<AGROW>
        m = mean(Y, 2); se = std(Y, 0, 2) / sqrt(size(Y, 2));
        plot(ax, X, m, '-', 'Color', ser{k, 2}, 'LineWidth', 1.3);
        for i = 1:numel(X), plot(ax, [X(i) X(i)], m(i) + [-1 1] * se(i), '-', 'Color', ser{k, 2}, 'LineWidth', 0.9); end
    end
    if c == 4
        if isempty(D.S1.cur_m18c_dec_empBig), text(ax, 15.6, 3.2, 'decoder not run', 'Color', RED, 'FontSize', FS, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle'); end
    end
    if YM(c) == 8, yt = 0:2:8; else, yt = 0:1:3; end
    set(ax, 'XLim', [0 16], 'XTick', 0:5:15, 'YLim', [-0.25 * YM(c) / 8 YM(c)], 'YTick', yt);
    xlabel(ax, 'Training data (h)');
    if any(c == [1 5]), ylabel(ax, 'Evidence (bits)'); end
    title(ax, TTL{c}, 'FontWeight', 'normal', 'FontSize', 8);
    letter(ax, LET{c}, -0.2);
end
% group header over E and F
hx = axes(fig, 'Units', 'centimeters', 'Position', [1.25 3.8 7.25 0.35], 'XLim', [0 1], 'YLim', [0 1], 'Visible', 'off'); hold(hx, 'on');
plot(hx, [0.02 0.98], [-0.12 -0.12], '-', 'Color', GREY, 'LineWidth', 0.5);
text(hx, 0.5, 0.62, 'Simulated ultrasound, optimistic physics', 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% ---------------- G: training needed to reach the goal (linear hours; band = not reached with all 15 h)
ax = axes(fig, 'Units', 'centimeters', 'Position', [9.85 1.15 3.45 Hh]); hold(ax, 'on');
CD = {'native', 'm6c', 'm12c', 'm18c', 'usL90', 'usB90'};
XL1 = {'0 dB', '-6', '-12', '-18', '1', '2'}; XL2 = {'', '\times4', '\times16', '\times63', '', ''};   % dB first, x as secondary (Omer 2026-10-21)
YB = 17.6; JIT = [-0.6 0 0.6];
fill(ax, [0.5 6.5 6.5 0.5], [16 16 19.2 19.2], [0.90 0.90 0.90], 'EdgeColor', 'none');
for k = 1:3
    M = nan(3, numel(CD)); off = (k - 2) * 0.17;
    for s = 1:3
        for c = 1:numel(CD)
            f = sprintf('hours_%s_%s', CD{c}, ser{k, 1}); o = D.(SUBJ{s}); if ~isfield(o, f), continue; end
            v = o.(f); kd = o.([f '_kind']); if iscell(v), v = cellfun(@numornan, v); end
            switch kd
                case {'point', 'below'}, y = v;
                case 'above', y = YB + JIT(s);
                otherwise, continue;
            end
            M(s, c) = y; plot(ax, c + off, y, 'o', 'MarkerSize', 3.4, 'MarkerFaceColor', ser{k, 2}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3);
        end
    end
    Mm = M; Mm(Mm > 16) = YB; m = median(Mm, 1);
    for g = {1:4, 5:6}
        ii = g{1}; ii = ii(~isnan(m(ii)));
        if numel(ii) > 1, plot(ax, ii + off, m(ii), '-', 'Color', ser{k, 2}, 'LineWidth', 1.2); end
    end
end
plot(ax, [4.5 4.5], [0 19.2], '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.6);
set(ax, 'XLim', [0.5 6.5], 'XTick', 1:6, 'XTickLabel', [], 'YLim', [0 19.2], 'YTick', [0:3:15 YB], ...
    'YTickLabel', {'0', '3', '6', '9', '12', '15', '>15'}, 'TickLabelInterpreter', 'tex');
for c = 1:6
    text(ax, c, -0.9, XL1{c}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS);
    text(ax, c, -3.4, XL2{c}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS, 'Color', GREY);
end
text(ax, 2.5, -6.2, 'Uniform noise', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS + 0.5);
text(ax, 5.5, -3.4, 'probes', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS, 'Color', GREY);
text(ax, 5.5, -6.2, 'Ultrasound', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS + 0.5);
ylabel(ax, 'Training needed (h)');
title(ax, 'Training needed', 'FontWeight', 'normal', 'FontSize', 8);
letter(ax, 'G', -0.33);

% ---------------- H: noise survived with the same training (uniform-noise ladder), per participant and median
ax = axes(fig, 'Units', 'centimeters', 'Position', [14.3 1.15 3.15 Hh]); hold(ax, 'on'); h0 = arrayfun(@numornan, D.S1.rung_hours);
for k = 1:3
    M = nan(3, numel(h0));
    for s = 1:3
        o = D.(SUBJ{s}); v = arrayfun(@numornan, o.(['ns_' ser{k, 1}])); kd = o.(['ns_' ser{k, 1} '_kind']);
        for i = 1:numel(h0)
            if isnan(h0(i)) || isnan(v(i)), continue; end
            switch kd{i}
                case 'point', y = -v(i); plot(ax, h0(i), y, 'o', 'MarkerSize', 3.0, 'MarkerFaceColor', ser{k, 2}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3); M(s, i) = y;
                case 'below', plot(ax, h0(i), 19.6, '^', 'MarkerSize', 3.0, 'MarkerFaceColor', ser{k, 2}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3); M(s, i) = 18;
                case 'above', plot(ax, h0(i), -1.6, 'v', 'MarkerSize', 3.0, 'MarkerFaceColor', ser{k, 2}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.3); M(s, i) = 0;
            end
        end
    end
    m = median(M, 1); ok = ~isnan(m) & sum(~isnan(M), 1) == 3;
    plot(ax, h0(ok), m(ok), '-', 'Color', ser{k, 2}, 'LineWidth', 1.2);
end
plot(ax, [0 16], [0 0], ':', 'Color', INK, 'LineWidth', 0.8);
set(ax, 'XLim', [0 16], 'XTick', 0:5:15, 'YLim', [-2.6 20.6], 'YTick', 0:6:18, 'YTickLabel', []);
YT1 = {'none', '-6 dB', '-12 dB', '-18 dB'}; YT2 = {'', '\times4', '\times16', '\times63'};   % dB first, x as secondary
for i = 1:4
    if i == 1
        text(ax, -0.9, 0, YT1{i}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', FS);
    else
        text(ax, -0.9, (i - 1) * 6 + 1.3, YT1{i}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', FS);
        text(ax, -0.9, (i - 1) * 6 - 1.5, YT2{i}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', FS, 'Color', GREY);
    end
end
xlabel(ax, 'Training data (h)');
title(ax, 'Uniform noise survived', 'FontWeight', 'normal', 'FontSize', 8);
letter(ax, 'H', -0.3);

exportgraphics(fig, [P 'figs/fig2_learning_e4.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, [P 'figs/fig2_learning_e4.png'], 'Resolution', 300, 'BackgroundColor', 'white'); close(fig);
disp('fig2 done');

function letter(ax, L, xo)
    text(ax, xo, 1.04, L, 'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'normal', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
end
function v = numornan(x)
    if isempty(x), v = nan; elseif iscell(x), v = cellfun(@numornan, x); else, v = double(x); end
end
