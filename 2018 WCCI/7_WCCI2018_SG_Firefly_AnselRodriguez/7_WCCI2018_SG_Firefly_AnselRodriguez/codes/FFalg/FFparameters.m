ffParameters.nPop= 10; % population in DE
ffParameters.gamma=1; %Mutation factor
ffParameters.beta0= 2; %Recombination constant
ffParameters.alpha=0.2;
ffParameters.alpha_damp=0.98;
ffParameters.m=2;

ffParameters.MaxIt= 500; % number of max iterations/gen
ffParameters.I_strategy   = 1; %DE strategy
ffParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back


% MaxIt=1000;         % Maximum Number of Iterations
% 
% nPop=25;            % Number of Fireflies (Swarm Size)
% 
% gamma=1;            % Light Absorption Coefficient
% 
% beta0=2;            % Attraction Coefficient Base Value
% 
% alpha=0.2;          % Mutation Coefficient
% 
% alpha_damp=0.98;    % Mutation Coefficient Damping Ratio
% 
% delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range
% 
% m=2;