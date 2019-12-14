mat = readmatrix('BLSPrices.csv');
tempVal = zeros(length(mat) - 2, size(mat, 2));

stock_Price = 100;
r = 0.05;

for col = 1:size(mat, 2)
    T = mat(1, col) / 252;
    K = mat(2, col);
    for i=3:length(mat)    
        tempVal(i - 2, col) = blsimpv(stock_Price, K, r, T, mat(i, col));
    end
end

%% Normalize

%normalized_IV = normalize(tempVal, 1);
%% Create IV map
S = 100;
r = 0.05;

times = [30/252 60/252 90/252 180/252 270/252];
strikes = [90 95 100 105 110];
prices = [10.684
6.084
2.306
0.396
0.036
11.458
7.078
3.445
1.135
0.247
12.193
7.957
4.397
1.900
0.603
14.268
10.272
6.823
4.069
2.117
16.174
12.355
8.925
6.043
3.785]
prices = reshape(prices, [5, 5]);

tempVal = zeros(5, 5);

for col = 1:5
    T = times(col);
    for i=1:5    
        K = strikes(i);
        tempVal(i, col) = blsimpv(S, K, r, T, prices(i, col));
    end
end
%%

[X, Y] = meshgrid([30 60 90 180 270], [90 95 100 105 110]);

figure
surf(X, Y, tempVal)
xlabel('Days to Maturity')
ylabel('Strike')
zlabel('IV')
zlim([0.13, 0.18])
set(gca, 'ydir', 'reverse')

%% Run PCA on entire surface

[coeff_All, score_All, latent_All, tsquared_All, explained_All, mu_All] = pca(tempVal, 'centered', 'off');


%% Run PCA by fixing maturity

[coeff_FixM30, score_FixM30, latent_FixM30, tsquared_FixM30, explained_FixM30, mu_FixM30] = pca(tempVal(:, (1:5)), 'centered', 'off'); 
[coeff_FixM60, score_FixM60, latent_FixM60, tsquared_FixM60, explained_FixM60, mu_FixM60] = pca(tempVal(:, (6:10)), 'centered', 'off'); 
[coeff_FixM90, score_FixM90, latent_FixM90, tsquared_FixM90, explained_FixM90, mu_FixM90] = pca(tempVal(:, (11:15)), 'centered', 'off'); 
[coeff_FixM180, score_FixM180, latent_FixM180, tsquared_FixM180, explained_FixM180, mu_FixM180] = pca(tempVal(:, (16:20)), 'centered', 'off'); 
[coeff_FixM270, score_FixM270, latent_FixM270, tsquared_FixM270, explained_FixM270, mu_FixM270] = pca(tempVal(:, (21:25)), 'centered', 'off'); 

%% Run PCA by fixing strike

[coeff_FixK90, score_FixK90, latent_FixK90, tsquared_FixK90, explained_FixK90, mu_FixK90] = pca(tempVal(:, (1:5:21)), 'centered', 'off'); 
[coeff_FixK95, score_FixK95, latent_FixK95, tsquared_FixK95, explained_FixK95, mu_FixK95] = pca(tempVal(:, (2:5:22)), 'centered', 'off'); 
[coeff_FixK100, score_FixK100, latent_FixK100, tsquared_FixK100, explained_FixK100, mu_FixK100] = pca(tempVal(:, (3:5:23)), 'centered', 'off'); 
[coeff_FixK105, score_FixK105, latent_FixK105, tsquared_FixK105, explained_FixK105, mu_FixK105] = pca(tempVal(:, (4:5:24)), 'centered', 'off'); 
[coeff_FixK110, score_FixK110, latent_FixK110, tsquared_FixK110, explained_FixK110, mu_FixK110] = pca(tempVal(:, (5:5:25)), 'centered', 'off'); 

%% Some Visualizations
scatter(score_All(:, 1), score_All(:, 2), 20, 'blue', 'filled')
title('PCA on Entire IV Surface, Projection onto 1st Two PC')
figure

%%

scatter(score_FixM30(:, 1), score_FixM30(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM60(:, 1), score_FixM60(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM90(:, 1), score_FixM90(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM180(:, 1), score_FixM180(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM270(:, 1), score_FixM270(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

%%
scatter(score_FixK90(:, 1), score_FixK90(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK95(:, 1), score_FixK95(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK100(:, 1), score_FixK100(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK105(:, 1), score_FixK105(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK110(:, 1), score_FixK110(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure