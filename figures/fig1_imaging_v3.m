% ISBI 2027 Figure 1, third design (2026-10-20, Omer: "a good 3d plot showing the different components ... with lines
% explaining what each component is ... and one bar link budget below, in addition of course to C").
% A: Blender renders (fig1_3d/): participant S1's head cut along a plane through the beam (real T1 slice), probe on the
%    left temporal window, beam to a preset voxel 5 cm deep; enlarged core along the beam with the budget's layers to
%    scale in depth; leader lines name each component and its two-way loss.
% B: the link budget as one bar per stage (echo; speech response), from figdata/fig1_data_k2.json (linkbudget/).
% C: unchanged from fig1_imaging.m.
% Run from the repository root: matlab -batch "run('figures/fig1_imaging_v3.m')"   (add -licmode onlinelicensing for an online licence)
P = [fileparts(mfilename('fullpath')) filesep];   % the figures/ directory of this repository
if ~exist([P 'figs'], 'dir'), mkdir([P 'figs']); end
R3 = [P 'fig1_3d/'];
D = jsondecode(fileread([P 'figdata/fig1_data_k2.json'])); B5 = D.budget5;   % K = 2 budget (Phase E4)
FN = 'Liberation Sans'; FS = 7;
set(groot, 'defaultAxesFontName', FN, 'defaultTextFontName', FN, 'defaultAxesFontSize', FS, 'defaultTextFontSize', FS, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off', 'defaultAxesLineWidth', 0.6, 'defaultAxesTickLength', [0.012 0.012]);
INK = [0.15 0.15 0.15]; GREY = [0.45 0.45 0.45]; LGREY = [0.62 0.62 0.62];
C_SKIN = [0.86 0.58 0.44]; C_BONE = [0.93 0.89 0.78]; C_BONE2 = [0.82 0.77 0.64]; C_BRAIN = [0.72 0.66 0.68]; C_PROBE = [0.30 0.31 0.33];
C_LOSS = [0.86 0.42 0.18]; C_GAIN = [0.16 0.45 0.80]; C_ECHO = [0.55 0.55 0.58]; TOT3 = [0.30 0.30 0.30];
er = cellfun(@(r) r{2}, B5.echo_rows); fr = cellfun(@(r) r{2}, B5.func_rows); nat = B5.native_dB(:)';
fig = figure('Units', 'centimeters', 'Position', [1 1 8.6 11.66], 'Color', 'w');

% ================= A: renders
[Ic, oc] = loadcrop([R3 'context_render.png']); Ac = jsondecode(fileread([R3 'context_render.json']));
[Ik, ok_] = loadcrop([R3 'core_render.png']); Ak = jsondecode(fileread([R3 'core_render.json']));
hc = 3.3; wc = hc * size(Ic, 2) / size(Ic, 1); yc = 7.15;
axc = axes(fig, 'Units', 'centimeters', 'Position', [0.05 yc wc hc]); image(axc, Ic); axis(axc, 'image', 'off');
hk = 4.25; wk = hk * size(Ik, 2) / size(Ik, 1); xk = 0.05 + wc + 0.3; yk = 7.1;
axk = axes(fig, 'Units', 'centimeters', 'Position', [xk yk wk hk]); image(axk, Ik); axis(axk, 'image', 'off');
ov = axes(fig, 'Units', 'centimeters', 'Position', [0 0 8.6 11.66], 'XLim', [0 8.6], 'YLim', [0 11.66], 'Visible', 'off'); hold(ov, 'on');
c2f = @(a, o, I, x0, y0, w, h) [x0 + (a(1) - o(1)) / size(I, 2) * w, y0 + h - (a(2) - o(2)) / size(I, 1) * h];
pc = c2f(Ac.anchors.probe, oc, Ic, 0.05, yc, wc, hc); vc = c2f(Ac.anchors.voxel, oc, Ic, 0.05, yc, wc, hc);
k = @(nm) c2f(Ak.anchors.(nm), ok_, Ik, xk, yk, wk, hk);
plot(ov, [pc(1) k('top_left')*[1;0]], [pc(2) k('top_left')*[0;1]], '--', 'Color', LGREY, 'LineWidth', 0.5);
plot(ov, [vc(1) k('bottom_left')*[1;0]], [vc(2) k('bottom_left')*[0;1]], '--', 'Color', LGREY, 'LineWidth', 0.5);
text(ov, 0.1, yc - 0.03, 'S1, MRI through the beam', 'Color', GREY, 'FontSize', FS - 0.5, 'VerticalAlignment', 'top');
text(ov, 8.55, yk - 0.03, sprintf('two-way: the echo returns %.1f dB below the noise', -er(9)), 'Color', INK, 'FontSize', FS - 0.5, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
% labels with leader lines, right of the core
xl = xk + wk + 0.22;
LB = {'probe', 11.1, 'Probe, 20-mm aperture', sprintf('%.1f dB; aberration %.1f', er(7), er(6));
      'scalp', 10.25, 'Scalp and muscle, 5 mm', sprintf('%.1f dB', er(4));
      'bone', 9.4, 'Temporal bone, 2.9 mm', sprintf('%.1f dB; surfaces %.1f', er(2), er(3));
      'brain', 8.4, 'Brain, 4.2 cm', sprintf('%.1f dB; clutter %.1f', er(5), er(8));
      'voxel', 7.45, 'Voxel 5 cm deep', '27 mm^3, the target'};
for i = 1:size(LB, 1)
    a = k(LB{i, 1}); y = LB{i, 2};
    plot(ov, [a(1) xl - 0.06], [a(2) y - 0.05], '-', 'Color', INK, 'LineWidth', 0.45);
    plot(ov, a(1), a(2), '.', 'Color', INK, 'MarkerSize', 5);
    text(ov, xl, y + 0.12, LB{i, 3}, 'FontSize', FS, 'Color', INK, 'VerticalAlignment', 'middle');
    text(ov, xl, y - 0.19, LB{i, 4}, 'FontSize', FS, 'Color', C_LOSS * 0.9, 'VerticalAlignment', 'middle');
end
text(ov, 0.0, 11.5, 'A', 'FontSize', 10, 'VerticalAlignment', 'middle');

% ================= B: left-to-right waterfall of the speech signal in one 2-s scan (values rounded)
% 1 perfect echo averaged over one 2-s scan (+19.9 = 48 samples + 10 images - 7.0); 2 skull and brain: with the echo 9.9 dB
% below the noise only s/(1+s) = 9% of the measured power is blood (-20.7 = counted twice + saturation); 3 speech changes
% blood volume by 4.5% (+ physiological noise, -27.2); 4 only 26% is sentence-specific (+ smaller fMRI voxel, -13.5);
% 5 ultrasound = -41.5; 6 fMRI median voxel -18.2; gap = noise added, 22 to 24 dB.
if numel(fr) == 11, fr = [fr(1:3); fr(4) + fr(9); fr(5:8); fr(10:11)]; end   % K = 10 data folded to the K = 2 row layout
sb = 10 ^ (er(9) / 10); fb = sb / (1 + sb); st = fr(3) + fr(4); us = fr(10); fm = mean(nat);
STEPS = {'total', st, C_PROBE * 0.5 + 0.5, {'perfect', 'echo'};
         'step', fr(1) + fr(2), C_PROBE * 0.6 + 0.4 * [1 1 1], {'echo is', sprintf('%.0f%% blood', 100 * fb)};
         'step', fr(5) + fr(6), C_LOSS, {'response', 'is 4.5%'};
         'step', fr(8) + fr(9), C_LOSS * 0.55 + 0.45, {'26%', 'specific'};
         'total', us, TOT3, {'ultra-', 'sound'};
         'total', fm, C_GAIN, {'fMRI'}};
bx = axes(fig, 'Units', 'centimeters', 'Position', [1.25 4.55 6.95 1.95]); hold(bx, 'on');
yline(bx, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
run = 0; w = 0.62; n = size(STEPS, 1);
for i = 1:n
    v = STEPS{i, 2};
    if strcmp(STEPS{i, 1}, 'total'), a = 0; b = v; run = v; s = sprintf('%+.1f', v); if v < 0, s = sprintf('%.1f', v); end; fw = 'bold';
    else, a = run; b = run + v; run = b; s = sprintf('%.1f', v); fw = 'normal'; end
    rectangle(bx, 'Position', [i - w / 2 min(a, b) w abs(b - a)], 'FaceColor', STEPS{i, 3}, 'EdgeColor', 'none');
    if i < 4, plot(bx, [i + w / 2 i + 1 - w / 2], [run run], '-', 'Color', GREY, 'LineWidth', 0.4); end
    if b >= a, ty = b + 1.5; va = 'bottom'; else, ty = min(a, b) - 1.5; va = 'top'; end
    if strcmp(STEPS{i, 1}, 'total') && v < 0, ty = v - 1.5; va = 'top'; end
    text(bx, i, ty, s, 'HorizontalAlignment', 'center', 'VerticalAlignment', va, 'FontSize', FS, 'Color', INK, 'FontWeight', fw);
    text(bx, i, -57.5, STEPS{i, 4}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', FS - 1, 'Color', INK);
end
plot(bx, [4 + w / 2 5 - w / 2], [us us], '-', 'Color', GREY, 'LineWidth', 0.4);
% medians only (Omer): the Monte Carlo range of this voxel is not drawn
GAP = [0.70 0.12 0.10]; xa = 5.5;                                                  % the comparison the paper is about
plot(bx, [5 + w / 2 xa], [us us], ':', 'Color', GAP, 'LineWidth', 0.8); plot(bx, [xa 6 - w / 2], [fm fm], ':', 'Color', GAP, 'LineWidth', 0.8);
annotation_arrow(bx, [xa (us + fm) / 2], [xa fm], GAP); annotation_arrow(bx, [xa (us + fm) / 2], [xa us], GAP);
text(bx, 6 + w / 2 + 0.1, (us + fm) / 2, {'SNR change', sprintf('%.0f to %.0f dB', us - min(nat), us - max(nat))}, 'FontSize', FS, 'Color', GAP, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
text(bx, 8.0, 1.0, 'signal = noise', 'FontSize', FS - 1, 'Color', GREY, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
set(bx, 'XLim', [0.4 8.0], 'YLim', [-50 24], 'XTick', [], 'XColor', 'none', 'YTick', -40:20:20, 'Clipping', 'off');
ylabel(bx, 'dB, one 2-s scan', 'FontSize', FS - 0.5);
text(ov, 0.0, 6.72, 'B', 'FontSize', 10, 'VerticalAlignment', 'middle');

% ================= C: SNR change of the readouts' voxels against distance, -10log10(1+f_v), plotted negative (Omer 2026-10-21: dB everywhere)
dx = axes(fig, 'Units', 'centimeters', 'Position', [1.25 0.85 5.55 2.32]); hold(dx, 'on');
ctr = (D.bins_mm(1:end - 1) + 2) / 10; ctr = ctr(:); SUB = {'S1', 'S2', 'S3'};
CS = {'usL10', [0.62 0.62 0.62], 0.0, true; 'usL50', INK, 0.72, false; 'usL90', [0.50 0.50 0.50], 0.72, false};
for r = [-6 -12 -18], plot(dx, [0.6 8.6], [r r], ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.6); end
for r = [-6 -12 -18], text(dx, 8.7, r, sprintf('%d dB', r), 'Color', GREY, 'FontSize', FS, 'VerticalAlignment', 'middle'); end
text(dx, 8.7, -1.0, 'uniform noise', 'Color', GREY, 'FontSize', FS, 'VerticalAlignment', 'middle');
for kk = 1:3
    Q = nan(numel(ctr), 3, 3);
    for s = 1:3
        Bq = D.subjects.(SUB{s}).(CS{kk, 1}).binned;
        for i = 1:numel(ctr)
            if iscell(Bq), bq = Bq{i}; else, bq = Bq(i, :); end; if iscell(bq), bq = cellfun(@numornan, bq); end
            Q(i, :, s) = bq(1:3);
        end
    end
    q = -mean(Q, 3); q = q(:, [3 2 1]); okq = all(~isnan(Q(:, 2, :)), 3); col = CS{kk, 2};
    if CS{kk, 4}, plot(dx, ctr(okq), q(okq, 2), '--', 'Color', col, 'LineWidth', 1.0);
    else
        fc = col + (1 - col) * CS{kk, 3};
        patch(dx, [ctr(okq); flipud(ctr(okq))], [q(okq, 1); flipud(q(okq, 3))], fc, 'EdgeColor', 'none', 'FaceAlpha', 0.75);
        plot(dx, ctr(okq), q(okq, 2), '-', 'Color', col, 'LineWidth', 1.4);
    end
end
text(dx, 7.0, -49, 'median', 'Color', INK, 'HorizontalAlignment', 'center');
text(dx, 2.6, -2.6, 'optimistic', 'Color', [0.40 0.40 0.40], 'HorizontalAlignment', 'center');
text(dx, 3.3, -44.5, 'pessimistic', 'Color', [0.45 0.45 0.45], 'HorizontalAlignment', 'center');
set(dx, 'XLim', [0.6 8.6], 'XTick', 1:8, 'YLim', [-54 0], 'YTick', -48:12:0);
xlabel(dx, 'Distance from the probe (cm)'); ylabel(dx, 'SNR change (dB)');
text(dx, 0.62, 1.03, 'SNR change with simulated ultrasound, one probe', 'Units', 'normalized', 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(dx, -0.2, 1.0, 'C', 'Units', 'normalized', 'FontSize', 10, 'VerticalAlignment', 'bottom');

exportgraphics(fig, [P 'figs/fig1_imaging_v3.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, [P 'figs/fig1_imaging_v3.png'], 'Resolution', 300, 'BackgroundColor', 'white'); close(fig);
disp('fig1 v3 done');

function barrow(ax, x, segs, INK, FS)
% one bar: from the start value x, each segment a loss drawn leftwards, label centred inside
    for i = 1:size(segs, 1)
        v = segs{i, 1}; rectangle(ax, 'Position', [x + v 0 -v 1], 'FaceColor', segs{i, 2}, 'EdgeColor', 'w', 'LineWidth', 0.4);
        if ~isempty(segs{i, 3}), text(ax, x + v / 2, 0.5, segs{i, 3}, 'HorizontalAlignment', 'center', 'FontSize', FS - 0.5, 'Color', INK); end
        x = x + v;
    end
    plot(ax, [0 0], [-0.25 0], '-', 'Color', INK, 'LineWidth', 0.9); plot(ax, [0 0], [1 1.25], '-', 'Color', INK, 'LineWidth', 0.9);
end
function annotation_arrow(ax, p0, p1, col)
    plot(ax, [p0(1) p1(1)], [p0(2) p1(2)], '-', 'Color', col, 'LineWidth', 0.6);
    d = p1 - p0; xl = diff(get(ax, 'XLim')); yl = diff(get(ax, 'YLim')); pos = get(ax, 'Position');
    dc = [d(1) / xl * pos(3), d(2) / yl * pos(4)]; u = dc / norm(dc); nrm = [-u(2) u(1)]; L = 0.12; W = 0.05;
    tip = [p1(1) / xl * pos(3), p1(2) / yl * pos(4)];
    b1 = tip - L * u + W * nrm; b2 = tip - L * u - W * nrm; cv = @(q) [q(1) * xl / pos(3), q(2) * yl / pos(4)];
    t0 = cv(tip); c1 = cv(b1); c2 = cv(b2); patch(ax, [t0(1) c1(1) c2(1)], [t0(2) c1(2) c2(2)], col, 'EdgeColor', 'none');
end
function [I, o] = loadcrop(f)
    [I, ~, a] = imread(f); a = double(a) / 255; I = double(I) / 255;
    I = I .* a + (1 - a); r = find(any(a > 0.02, 2)); c = find(any(a > 0.02, 1)); pad = 6;
    r0 = max(r(1) - pad, 1); r1 = min(r(end) + pad, size(I, 1)); c0 = max(c(1) - pad, 1); c1 = min(c(end) + pad, size(I, 2));
    I = I(r0:r1, c0:c1, :); o = [c0 - 1, r0 - 1];
end
function v = numornan(x)
    if isempty(x), v = nan; elseif iscell(x), v = cellfun(@numornan, x); else, v = double(x); end
end
