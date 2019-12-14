matHeston = readmatrix('HestonPrices.csv');
tempValHeston = zeros(length(matHeston) - 2, size(matHeston, 2));

stock_Price = 100;
r = 0.05;

for col = 1:size(matHeston, 2)
    T = matHeston(1, col) / 252;
    K = matHeston(2, col);
    for i=3:length(matHeston)    
        tempValHeston(i - 2, col) = blsimpv(stock_Price, K, r, T, matHeston(i, col));
    end
end

%% Normalize

%normalized_IV = normalize(tempVal, 1);
%% Create Average IV map

averaged_IVsHeston = mean(tempValHeston, 1);

[X, Y] = meshgrid([30 60 90 180 270], [90 95 100 105 110]);

surf(X, Y, reshape(averaged_IVsHeston, [5, 5]))
xlabel('Days to Maturity')
ylabel('Strike')
zlabel('IV')
zlim([0.13, 0.18])
set(gca, 'ydir', 'reverse')

%% Run PCA on entire surface

[coeff_AllHeston, score_AllHeston, latent_AllHeston, tsquared_AllHeston, explained_AllHeston, mu_AllHeston] = pca(tempValHeston);


%% Run PCA by fixing maturity

[coeff_FixM30Heston, score_FixM30Heston, latent_FixM30Heston, tsquared_FixM30Heston, explained_FixM30Heston, mu_FixM30Heston] = pca(tempValHeston(:, (1:5))); 
[coeff_FixM60Heston, score_FixM60Heston, latent_FixM60Heston, tsquared_FixM60Heston, explained_FixM60Heston, mu_FixM60Heston] = pca(tempValHeston(:, (6:10))); 
[coeff_FixM90Heston, score_FixM90Heston, latent_FixM90Heston, tsquared_FixM90Heston, explained_FixM90Heston, mu_FixM90Heston] = pca(tempValHeston(:, (11:15))); 
[coeff_FixM180Heston, score_FixM180Heston, latent_FixM180Heston, tsquared_FixM180Heston, explained_FixM180Heston, mu_FixM180Heston] = pca(tempValHeston(:, (16:20))); 
[coeff_FixM270Heston, score_FixM270Heston, latent_FixM270Heston, tsquared_FixM270Heston, explained_FixM270Heston, mu_FixM270Heston] = pca(tempValHeston(:, (21:25))); 

%% Run PCA by fixing strike

[coeff_FixK90Heston, score_FixK90Heston, latent_FixK90Heston, tsquared_FixK90Heston, explained_FixK90Heston, mu_FixK90Heston] = pca(tempValHeston(:, (1:5:21))); 
[coeff_FixK95Heston, score_FixK95Heston, latent_FixK95Heston, tsquared_FixK95Heston, explained_FixK95Heston, mu_FixK95Heston] = pca(tempValHeston(:, (2:5:22))); 
[coeff_FixK100Heston, score_FixK100Heston, latent_FixK100Heston, tsquared_FixK100Heston, explained_FixK100Heston, mu_FixK100Heston] = pca(tempValHeston(:, (3:5:23))); 
[coeff_FixK105Heston, score_FixK105Heston, latent_FixK105Heston, tsquared_FixK105Heston, explained_FixK105Heston, mu_FixK105Heston] = pca(tempValHeston(:, (4:5:24))); 
[coeff_FixK110Heston, score_FixK110Heston, latent_FixK110Heston, tsquared_FixK110Heston, explained_FixK110Heston, mu_FixK110Heston] = pca(tempValHeston(:, (5:5:25))); 


%% Some Visualizations
scatter(score_AllHeston(:, 1), score_AllHeston(:, 2), 20, 'blue', 'filled')
title('PCA on Entire IV Surface, Projection onto 1st Two PC (Heston)')
figure

%%

scatter(score_FixM30Heston(:, 1), score_FixM30Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM60Heston(:, 1), score_FixM60Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM90Heston(:, 1), score_FixM90Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM180Heston(:, 1), score_FixM180Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM270Heston(:, 1), score_FixM270Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

%%
scatter(score_FixK90Heston(:, 1), score_FixK90Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK95Heston(:, 1), score_FixK95Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK100Heston(:, 1), score_FixK100Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK105Heston(:, 1), score_FixK105Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK110Heston(:, 1), score_FixK110Heston(:, 2), 20, 'blue', 'filled')
xlabel('1st PC')
ylabel('2nd PC')
figure
