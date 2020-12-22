
% deParameters.I_NP= 10; % population in DE
% deParameters.F_weight= 0.3; %Mutation factor
% deParameters.F_CR= 0.5; %Recombination constant
% deParameters.I_itermax= 499; % number of max iterations/gen
% deParameters.I_strategy   = 1; %DE strategy
% 
% deParameters.I_bnd_constr = 1; %Using bound constraints 

chaos_DEEPSO_parameters.I_NP=10; % 10 ok
chaos_DEEPSO_parameters.I_itermax= 280; % number of max iterations/gen, 277 ok
% levydeepso.mutationRate = 0.5;
%         levydeepso.communicationProbability = 0.5;
%         levydeepso.localSearchProbability = 0.25;
%         levydeepso.localSearchContinuousDiscrete = 0.75;
%         ff_par.excludeBranchViolations = 0;
%         
%             ff_par.factor = 1;
% levydeepso.maxIterations = 800; % number of max iterations/epochs
% levydeepso.fnc='fitnessFun_DER';
% levydeepso.noIterationsToGap=400; % iterations to wait for fit improve
% levydeepso.minIterations = 500; % number of min iterations/epochs
% levydeepso.threshold = 1e-9; % threshold for fitness improvement