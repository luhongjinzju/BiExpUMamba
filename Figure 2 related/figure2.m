clear all
close all
clc

%% ------------------ Initialization ------------------ %%
Path = "/Users/luhongjin/Desktop/Figures-lhj/Figure 2/0-matlab-segmentation/";
if exist(fullfile(Path, 'Intensity.xlsx'), 'file') ~= 2
    Path = [pwd filesep];
end

Intensity = xlsread(strcat(Path,'Intensity.xlsx'), "B2:CW101");
[m,n] = size(Intensity);
frame_interval = 0.2;   % 100 frames across 20 s for Figure 2
t = (0:m-1)' * frame_interval;

%% ------------------  Curve Fitting ----------------- %%
options = get_paper_method_options('bi');
options_one = get_paper_method_options('single');

Intensity_average = zeros(size(Intensity(:,1)));

for i = 1:n
    Int_4_fit = Intensity(:,i);
    Int_4_fit = Int_4_fit/max(Int_4_fit);
    Intensity_average = Intensity_average + Int_4_fit;

    [fitresult, gof] = robust_huber_lm_fit(t, Int_4_fit, options);

    Fitting_bi(i,1) = fitresult.I1; Fitting_bi(i,2) = fitresult.tau1; Fitting_bi(i,3) = fitresult.I2; Fitting_bi(i,4) = fitresult.tau2;
    Fitting_bi(i,5) = fitresult.Ibg;

    Fitting_bi(i,6)  = gof.sse; Fitting_bi(i,7)  = gof.rsquare; Fitting_bi(i,8)  = gof.dfe; Fitting_bi(i,9)  = gof.adjrsquare;
    Fitting_bi(i,10) = gof.rmse;

    [fitresult_one, gof_one] = robust_huber_lm_fit(t, Int_4_fit, options_one);

    Fitting_single(i,1) = fitresult_one.I1; Fitting_single(i,2) = fitresult_one.tau1;
    Fitting_single(i,3) = fitresult_one.Ibg;

    Fitting_single(i,4) = gof_one.sse; Fitting_single(i,5) = gof_one.rsquare; Fitting_single(i,6) = gof_one.dfe; Fitting_single(i,7) = gof_one.adjrsquare;
    Fitting_single(i,8) = gof_one.rmse;
end
strcat('bi-rsquare:',num2str(mean(Fitting_bi(:,7))))
strcat('bi-rmse:',   num2str(mean(Fitting_bi(:,10))))

strcat('single-rsquare:',num2str(mean(Fitting_single(:,5))))
strcat('single-rmse:',   num2str(mean(Fitting_single(:,8))))

Intensity_average = Intensity_average/max(Intensity_average);
[fitresult_bi,     gof_bi]     = robust_huber_lm_fit(t, Intensity_average, options);
[fitresult_single, gof_single] = robust_huber_lm_fit(t, Intensity_average, options_one);

strcat('bi-all-rsquare:',num2str(gof_bi.rsquare))
strcat('bi-all-rmse:',   num2str(gof_bi.rmse))

strcat('single-all-rsquare:',num2str(gof_single.rsquare))
strcat('single-all-rmse:',   num2str(gof_single.rmse))


figure;hold on
plot(t, Intensity_average, 'go');
plot(t, bi_exp_eval(fitresult_bi, t),   'b-');
plot(t, single_exp_eval(fitresult_single, t),  'r-');
legend('Real','Bi-exp','Single_exp');
xlabel('Timelapse (s)');
ylabel('Normalized Intensity (a.u.)');


xlswrite(fullfile(Path,'Fitting_single.xlsx'), Fitting_single);
xlswrite(fullfile(Path,'Fitting_bi.xlsx'), Fitting_bi);


function options = get_paper_method_options(model_type)
options.model_type = model_type;
options.delta_huber = 0.03;
options.max_irls_iter = 20;
options.irls_tol = 1e-6;

if strcmp(model_type, 'bi')
    options.startpoint = [0.65 2 0.35 10 0];
else
    options.startpoint = [1 5 0];
end
end


function [fitresult, gof] = robust_huber_lm_fit(x, y, options)
params = get_initial_guess(x, y, options);
weights = ones(size(y));
prev_params = inf(size(params));
prev_weights = weights;

for k = 1:options.max_irls_iter
    params = lm_iteration(params, x, y, options, weights);

    if strcmp(options.model_type, 'bi')
        params = reorder_bi_params(params);
    end

    residual = y - model_eval(params, x, options.model_type);
    weights = ones(size(residual));
    idx = abs(residual) > options.delta_huber;
    weights(idx) = options.delta_huber ./ abs(residual(idx));

    param_change = max(abs(params - prev_params) ./ max(abs(prev_params), 1));
    weight_change = max(abs(weights - prev_weights));
    if param_change < options.irls_tol && weight_change < options.irls_tol
        break
    end

    prev_params = params;
    prev_weights = weights;
end

fitted = model_eval(params, x, options.model_type);
gof = calculate_gof(y, fitted, numel(params));

if strcmp(options.model_type, 'bi')
    fitresult.I1 = params(1); fitresult.tau1 = params(2); fitresult.I2 = params(3); fitresult.tau2 = params(4); fitresult.Ibg = params(5);
else
    fitresult.I1 = params(1); fitresult.tau1 = params(2); fitresult.Ibg = params(3);
end
end


function params = lm_iteration(params0, x, y, options, weights)
params = params0(:).';
lambda = 1e-2;
lambda_max = 1e12;
lm_tol = 1e-8;
max_lm_iter = 40;

current_obj = objective_value(params, x, y, options, weights);

for iter = 1:max_lm_iter
    [weighted_residual, jacobian] = weighted_residual_and_jacobian(params, x, y, options, weights);
    gradient = jacobian' * weighted_residual;
    if norm(gradient, inf) < lm_tol
        break
    end

    hessian_approx = jacobian' * jacobian;
    damping_matrix = diag(max(diag(hessian_approx), 1e-8));
    linear_system = hessian_approx + lambda * damping_matrix;

    if rcond(linear_system) < 1e-12
        step = -pinv(linear_system) * gradient;
    else
        step = -linear_system \ gradient;
    end

    if any(~isfinite(step))
        lambda = min(lambda * 10, lambda_max);
        continue
    end

    candidate = params + step.';
    candidate_obj = objective_value(candidate, x, y, options, weights);

    if candidate_obj < current_obj
        params = candidate;
        current_obj = candidate_obj;
        lambda = max(lambda / 3, 1e-7);
        if norm(step) < lm_tol * (norm(params) + lm_tol)
            break
        end
    else
        lambda = min(lambda * 5, lambda_max);
    end
end
end


function [weighted_residual, jacobian] = weighted_residual_and_jacobian(params, x, y, options, weights)
base_residual = residual_vector(params, x, y, options);
sqrt_weights = sqrt(weights(:));
weighted_residual = sqrt_weights .* base_residual(:);

num_points = numel(y);
num_params = numel(params);
jacobian = zeros(num_points, num_params);

for j = 1:num_params
    step_size = 1e-6 * max(abs(params(j)), 1);
    trial_params = params;
    trial_params(j) = trial_params(j) + step_size;

    trial_residual = residual_vector(trial_params, x, y, options);
    trial_weighted_residual = sqrt_weights .* trial_residual(:);
    jacobian(:, j) = (trial_weighted_residual - weighted_residual) / step_size;
end
end


function value = objective_value(params, x, y, options, weights)
residual = residual_vector(params, x, y, options);
if any(~isfinite(residual))
    value = inf;
    return
end
weighted_residual = sqrt(weights(:)) .* residual(:);
value = 0.5 * sum(weighted_residual .^ 2);
end


function residual = residual_vector(params, x, y, options)
if ~is_valid_params(params, options.model_type)
    residual = 1e6 * ones(size(y));
    return
end
residual = y - model_eval(params, x, options.model_type);
end


function valid = is_valid_params(params, model_type)
if strcmp(model_type, 'bi')
    valid = all(params(1:4) > 0);
else
    valid = all(params(1:2) > 0);
end
end


function params = get_initial_guess(x, y, options)
baseline0 = min(max(y(end), -0.05), 0.5);

if strcmp(options.model_type, 'bi')
    params = options.startpoint;
    params(1) = max(params(1) * max(y(1) - baseline0, 0.2), 0.1);
    params(3) = max(params(3) * max(y(1) - baseline0, 0.2), 0.05);
    params(5) = baseline0;
else
    params = options.startpoint;
    params(1) = max(y(1) - baseline0, 0.2);
    params(3) = baseline0;
end
end


function params = reorder_bi_params(params)
if params(2) > params(4)
    params = [params(3), params(4), params(1), params(2), params(5)];
end
end


function yfit = model_eval(params, x, model_type)
if strcmp(model_type, 'bi')
    yfit = params(1) * exp(-x ./ params(2)) + params(3) * exp(-x ./ params(4)) + params(5);
else
    yfit = params(1) * exp(-x ./ params(2)) + params(3);
end
end


function yfit = bi_exp_eval(fitresult, x)
yfit = fitresult.I1 * exp(-x ./ fitresult.tau1) + fitresult.I2 * exp(-x ./ fitresult.tau2) + fitresult.Ibg;
end


function yfit = single_exp_eval(fitresult, x)
yfit = fitresult.I1 * exp(-x ./ fitresult.tau1) + fitresult.Ibg;
end


function gof = calculate_gof(y, fitted, num_params)
residual = y - fitted;
sse = sum(residual .^ 2);
sst = sum((y - mean(y)) .^ 2);

if sst <= eps
    rsquare = 1;
else
    rsquare = 1 - sse / sst;
end

dfe = max(numel(y) - num_params, 0);
if numel(y) - num_params - 1 > 0
    adjrsquare = 1 - (1 - rsquare) * (numel(y) - 1) / (numel(y) - num_params - 1);
else
    adjrsquare = rsquare;
end

if dfe > 0
    rmse = sqrt(sse / dfe);
else
    rmse = sqrt(mean(residual .^ 2));
end

gof.sse = sse;
gof.rsquare = rsquare;
gof.dfe = dfe;
gof.adjrsquare = adjrsquare;
gof.rmse = rmse;
end
