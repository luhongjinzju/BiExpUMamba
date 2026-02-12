clear all
close all
clc

%% ------------------ Initialization ------------------ %%
Path = "/Users/luhongjin/Desktop/HZNU/研究生相关材料/1-软件工程-邬沛宏/小论文1/Figures-lhj/Figure 4 相关/";
Intensity_structure  = xlsread(strcat(Path,'Intensity.xlsx'),1,"B2:AE101");
Intensity_bg_inside  = xlsread(strcat(Path,'Intensity.xlsx'),2,"B2:AE101");
Intensity_bg_outside = xlsread(strcat(Path,'Intensity.xlsx'),3,"B2:AE101");
[m,n] = size(Intensity_structure);

%% ------------------  Curve Fitting ----------------- %%

ft = fittype('a*exp(-b*x)+c*exp(-d*x)+e', 'independent', 'x', 'coefficients', {'a','b','c','d','e'});
options = fitoptions('Method', 'NonlinearLeastSquares', 'Startpoint', [1 0.01 0.5 0.001 -0.001], 'Lower',[0.0000001,0.0000001,0.0000001,0.0000001,-100]);

Intensity_structure  = Intensity_structure./max(Intensity_structure);
Intensity_bg_inside  = Intensity_bg_inside./max(Intensity_bg_inside);
Intensity_bg_outside = Intensity_bg_outside./max(Intensity_bg_outside);

Intensity_structure  = mean(Intensity_structure');  Intensity_structure  = Intensity_structure';
Intensity_bg_inside  = mean(Intensity_bg_inside');  Intensity_bg_inside  = Intensity_bg_inside';
Intensity_bg_outside = mean(Intensity_bg_outside'); Intensity_bg_outside = Intensity_bg_outside';

x = 1:m;

Intensity_structure  = Intensity_structure/max(Intensity_structure);
Intensity_bg_inside  = Intensity_bg_inside/max(Intensity_bg_inside);
Intensity_bg_outside = Intensity_bg_outside/max(Intensity_bg_outside);

[fitresult_str,    gof_str]    = fit(x', Intensity_structure, ft, options);
[fitresult_bg_in,  gof_bg_in]  = fit(x', Intensity_bg_inside, ft, options);
[fitresult_bg_out, gof_bg_out] = fit(x', Intensity_bg_outside, ft, options);

figure;hold on
plot(x, Intensity_structure, 'g*'); plot(x, fitresult_str(x),   'g-'); A = fitresult_str(x);
plot(x, Intensity_bg_inside, 'ro'); plot(x, fitresult_bg_in(x), 'r-'); B = fitresult_bg_in(x);
plot(x, Intensity_bg_outside,'b*'); plot(x, fitresult_bg_out(x),'b-'); C = fitresult_bg_out(x);


legend('Structure','Structure','bg_inside','bg_inside','bg_outside','bg_inside');
xlabel('Timelapse');
ylabel('Normalized Intensity');

fit = fitresult_str; gof = gof_str;
R(1,1) = fit.a;   R(1,2) = fit.b;       R(1,3) = fit.c;   R(1,4) = fit.d;          R(1,5) = fit.e;   
R(1,6) = gof.sse; R(1,7) = gof.rsquare; R(1,8) = gof.dfe; R(1,9) = gof.adjrsquare; R(1,10) = gof.rmse; 
xlswrite('fitresult_str.xlsx', R);

fit = fitresult_bg_in; gof = gof_bg_in;
R(1,1) = fit.a;   R(1,2) = fit.b;       R(1,3) = fit.c;   R(1,4) = fit.d;          R(1,5) = fit.e;   
R(1,6) = gof.sse; R(1,7) = gof.rsquare; R(1,8) = gof.dfe; R(1,9) = gof.adjrsquare; R(1,10) = gof.rmse; 
xlswrite('fitresult_bg_in.xlsx', R);

fit = fitresult_bg_out; gof = gof_bg_out;
R(1,1) = fit.a;   R(1,2) = fit.b;       R(1,3) = fit.c;   R(1,4) = fit.d;          R(1,5) = fit.e;   
R(1,6) = gof.sse; R(1,7) = gof.rsquare; R(1,8) = gof.dfe; R(1,9) = gof.adjrsquare; R(1,10) = gof.rmse; 
xlswrite('fitresult_bg_out.xlsx', R);
 
