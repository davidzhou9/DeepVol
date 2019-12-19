%% parameters for integration


m_f = -1.3;
m_s = -1;
Y_0 = -0.949;
Z_0 = -0.949;
rho_12 = 0.35;
alpha = 20;
delta = 0.1;
v_f = 0.5;
v_s = 0.8;

T = 60/252;

epsilon = 10^(-5);
currIntegralValue = (1 / T) * integral(@(x)fun(x, m_f, m_s, Y_0, Z_0, rho_12, alpha, delta, v_f, v_s), 0, T);
target = 0.0225;
%%

rhoArry = 0:0.001:0.9;

bestRho = -1;
bestRhoVal = 100;
%%

for i=1:length(rhoArry)
   tempVal = (1 / T) * integral(@(x)fun(x, m_f, m_s, Y_0, Z_0, rhoArry(i), alpha, delta, v_f, v_s), 0, T);
   if abs(bestRhoVal - target) > abs(tempVal - target)
       bestRho = rhoArry(i);
       bestRhoVal = tempVal;
   end
end    

%%
num_Steps = 0;

while (abs(currIntegralValue - target) > epsilon && num_Steps < 50000)
    if currIntegralValue > target
        rho_12 = (rho_12 - 1) / 2;
    else
        rho_12 = (rho_12 + 1) / 2;
    end
    currIntegralValue = (1 / T) * integral(@(x)fun(x, m_f, m_s, Y_0, Z_0, rho_12, alpha, delta, v_f, v_s), 0, T);
    num_Steps = num_Steps + 1;
end

%%
fun = @(x, m_f, m_s, Y_0, Z_0, rho_12, alpha, delta, v_f, v_s) exp(2 * (m_f + (Y_0 - m_f) * exp(-alpha * x) + m_s + (Z_0 - m_s) * exp(-delta * x) + v_f^2 * (1 - exp(-2 * alpha * x)) + v_s^2 * (1 - exp(-2 * delta * x)) + ((4 * v_f * v_s * rho_12 * sqrt(alpha * delta)) / (alpha + delta)) * (1 - exp(-x * (alpha + delta)))));


fun2 = @(x, c) c * x;