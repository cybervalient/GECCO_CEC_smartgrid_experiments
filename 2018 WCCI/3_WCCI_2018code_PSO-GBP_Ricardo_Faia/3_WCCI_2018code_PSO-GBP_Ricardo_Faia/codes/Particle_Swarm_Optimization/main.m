clear;
clc;
close all;
psoParameters
for i=1:1
[BestSol]=pso(PSOparameters);
results(i,1)=(BestSol.Best);
end
format long;
[mean(results) min(results) std(results)]