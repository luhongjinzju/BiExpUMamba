clear all
close all
clc

%% ------------------ Initialization ------------------ %%
Path = "/Users/luhongjin/Desktop/Figures-lhj/Figure 2/0-matlab-segmentation/";
Intensity = xlsread(strcat(Path,'Intensity.xlsx'),"B2:CW101");
[m,n] = size(Intensity);

%% ------------------  Curve Fitting ----------------- %%

ft = fittype('a*exp(-b*x)+c*exp(-d*x)+e', 'independent', 'x', 'coefficients', {'a','b','c','d','e'});
options = fitoptions('Method', 'NonlinearLeastSquares', 'Startpoint', [1 0.01 0.5 0.001 -0.001], 'Lower',[0.0000001,0.0000001,0.0000001,0.0000001,-100]);

ft_one = fittype('a*exp(-b*x)+c', 'independent', 'x', 'coefficients', {'a','b','c'});
options_one = fitoptions('Method', 'NonlinearLeastSquares', 'Startpoint', [1 0.01 -0.001],'Lower',[0.0000001,0.0000001,-100]);

Intensity_average = zeros(size(Intensity(:,1)));

x = 1:m;
for i = 1:n
    Int_4_fit = Intensity(:,i);
    Int_4_fit = Int_4_fit/max(Int_4_fit);
    Intensity_average = Intensity_average + Int_4_fit;

    [fitresult, gof] = fit(x', Int_4_fit, ft,options);
     
    Fitting_bi(i,1) = fitresult.a; Fitting_bi(i,2) = fitresult.b; Fitting_bi(i,3) = fitresult.c; Fitting_bi(i,4) = fitresult.d;
    Fitting_bi(i,5) = fitresult.e;

    Fitting_bi(i,6)  = gof.sse; Fitting_bi(i,7)  = gof.rsquare; Fitting_bi(i,8)  = gof.dfe; Fitting_bi(i,9)  = gof.adjrsquare;
    Fitting_bi(i,10) = gof.rmse;
    
    [fitresult_one, gof_one] = fit(x', Int_4_fit, ft_one,options_one);
     
    Fitting_single(i,1) = fitresult_one.a; Fitting_single(i,2) = fitresult_one.b;
    Fitting_single(i,3) = fitresult_one.c;

    Fitting_single(i,4) = gof_one.sse; Fitting_single(i,5) = gof_one.rsquare; Fitting_single(i,6) = gof_one.dfe; Fitting_single(i,7) = gof_one.adjrsquare;
    Fitting_single(i,8) = gof_one.rmse;
end
strcat('bi-rsquare:',num2str(mean(Fitting_bi(:,7))))
strcat('bi-rmse:',   num2str(mean(Fitting_bi(:,10))))

strcat('single-rsquare:',num2str(mean(Fitting_single(:,5))))
strcat('single-rmse:',   num2str(mean(Fitting_single(:,8))))

Intensity_average = Intensity_average/max(Intensity_average);
[fitresult_bi,     gof_bi]     = fit(x', Intensity_average, ft,     options);
[fitresult_single, gof_single] = fit(x', Intensity_average, ft_one, options_one);

strcat('bi-all-rsquare:',num2str(gof_bi.rsquare))
strcat('bi-all-rmse:',   num2str(gof_bi.rmse))

strcat('single-all-rsquare:',num2str(gof_single.rsquare))
strcat('single-all-rmse:',   num2str(gof_single.rmse))


figure;hold on
plot(Intensity_average, x, 'g*');
plot(fitresult_bi(x),   x, 'b-'); 
plot(fitresult_one(x),  x, 'r-');
legend('Real','Bi-exp','Single_exp');
xlabel('Timelapse');
ylabel('Normalized Intensity');


xlswrite('Fitting_single.xlsx', Fitting_single);
xlswrite('Fitting_bi.xlsx', Fitting_bi);
