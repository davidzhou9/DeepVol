matMS = readmatrix('MSPrices.csv');
tempValMS = zeros(length(matMS) - 2, size(matMS, 2));

stock_Price = 100;
r = 0.05;

for col = 1:size(matMS, 2)
    T = matMS(1, col) / 252;
    K = matMS(2, col);
    for i=3:length(matMS)    
        tempValMS(i - 2, col) = blsimpv(stock_Price, K, r, T, matMS(i, col));
    end
end

%% Create Average IV map

averaged_IVsMS = mean(tempValMS, 1);

[X, Y] = meshgrid([30 60 90 180 270], [90 95 100 105 110]);

surf(X, Y, reshape(averaged_IVsMS, [5, 5]))
xlabel('Days to Maturity')
ylabel('Strike')
zlabel('IV')
zlim([0.13, 0.18])
set(gca, 'ydir', 'reverse')

%% Run PCA on entire surface

[coeff_AllMS, score_AllMS, latent_AllMS, tsquared_AllMS, explained_AllMS, mu_AllMS] = pca(tempValMS);


%% Perform factor loading analysis on entire surface
%MSAll_FactorLoadings = zeros(length(coeff_AllMS), length(coeff_AllMS));

%for i = 1:length(coeff_AllMS)
%    if i == 1
%        MSAll_FactorLoadings(:, i) = coeff_AllMS(:, 1).^2;
%    else
%        MSAll_FactorLoadings(:, i) = coeff_AllMS(:, i).^2 + MSAll_FactorLoadings(:, i - 1);
%    end
%end

%% Run PCA by fixing maturity

[coeff_FixM30MS, score_FixM30MS, latent_FixM30MS, tsquared_FixM30MS, explained_FixM30MS, mu_FixM30MS] = pca(tempValMS(:, (1:5))); 
[coeff_FixM60MS, score_FixM60MS, latent_FixM60MS, tsquared_FixM60MS, explained_FixM60MS, mu_FixM60MS] = pca(tempValMS(:, (6:10))); 
[coeff_FixM90MS, score_FixM90MS, latent_FixM90MS, tsquared_FixM90MS, explained_FixM90MS, mu_FixM90MS] = pca(tempValMS(:, (11:15))); 
[coeff_FixM180MS, score_FixM180MS, latent_FixM180MS, tsquared_FixM180MS, explained_FixM180MS, mu_FixM180MS] = pca(tempValMS(:, (16:20))); 
[coeff_FixM270MS, score_FixM270MS, latent_FixM270MS, tsquared_FixM270MS, explained_FixM270MS, mu_FixM270MS] = pca(tempValMS(:, (21:25))); 

%% Perform factor analysis, fixed maturity

%MSM60_FactorLoadings = zeros(length(coeff_FixM30MS), length(coeff_FixM30MS));

%for i = 1:length(coeff_FixM30MS)
    
    %if i == 1
    %    MSM60_FactorLoadings(:, i) = coeff_FixM60MS(:, 1).^2;
    %else
    %    MSM60_FactorLoadings(:, i) = coeff_FixM60MS(:, i).^2 + MSM60_FactorLoadings(:, i - 1);
    %end
    %MSM30_FactorLoadings(:, i) = coeff_FixM30MS(:, i).^2;
%end

%% Run PCA by fixing strike

[coeff_FixK90MS, score_FixK90MS, latent_FixK90MS, tsquared_FixK90MS, explained_FixK90MS, mu_FixK90MS] = pca(tempValMS(:, (1:5:21))); 
[coeff_FixK95MS, score_FixK95MS, latent_FixK95MS, tsquared_FixK95MS, explained_FixK95MS, mu_FixK95MS] = pca(tempValMS(:, (2:5:22))); 
[coeff_FixK100MS, score_FixK100MS, latent_FixK100MS, tsquared_FixK100MS, explained_FixK100MS, mu_FixK100MS] = pca(tempValMS(:, (3:5:23))); 
[coeff_FixK105MS, score_FixK105MS, latent_FixK105MS, tsquared_FixK105MS, explained_FixK105MS, mu_FixK105MS] = pca(tempValMS(:, (4:5:24))); 
[coeff_FixK110MS, score_FixK110MS, latent_FixK110MS, tsquared_FixK110MS, explained_FixK110MS, mu_FixK110MS] = pca(tempValMS(:, (5:5:25))); 

%% Some Visualizations
scatter(score_AllMS(:, 1), score_AllMS(:, 2), 20, 'blue', 'filled')
title('PCA on Entire IV Surface, Projection onto 1st Two PC (MS)')
figure

%%

scatter(score_FixM30MS(:, 1), score_FixM30MS(:, 2), 20, 'blue', 'filled')
title('PCA for T = 30, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM60MS(:, 1), score_FixM60MS(:, 2), 20, 'blue', 'filled')
title('PCA for T = 60, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM90MS(:, 1), score_FixM90MS(:, 2), 20, 'blue', 'filled')
title('PCA for T = 90, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM180MS(:, 1), score_FixM180MS(:, 2), 20, 'blue', 'filled')
title('PCA for T = 180, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixM270MS(:, 1), score_FixM270MS(:, 2), 20, 'blue', 'filled')
title('PCA for T = 270, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

%%
scatter(score_FixK90MS(:, 1), score_FixK90MS(:, 2), 20, 'blue', 'filled')
title('PCA for K = 90, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK95MS(:, 1), score_FixK95MS(:, 2), 20, 'blue', 'filled')
title('PCA for K = 95, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK100MS(:, 1), score_FixK100MS(:, 2), 20, 'blue', 'filled')
title('PCA for K = 100, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK105MS(:, 1), score_FixK105MS(:, 2), 20, 'blue', 'filled')
title('PCA for K = 105, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure

scatter(score_FixK110MS(:, 1), score_FixK110MS(:, 2), 20, 'blue', 'filled')
title('PCA for K = 110, Projection onto First Two PCs (MS)')
xlabel('1st PC')
ylabel('2nd PC')
figure
