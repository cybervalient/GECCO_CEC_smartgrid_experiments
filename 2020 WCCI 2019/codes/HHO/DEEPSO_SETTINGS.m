%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function [Deepso,Ff,Set] = DEEPSO_SETTINGS(low_habitat_limit,up_habitat_limit,deParameters,gbest)

D=size(low_habitat_limit);
Set.D            = D(1,2);                   % Variables to optimize

Set.Xmax         = up_habitat_limit;    % Individuals' lower bounds.
Set.Xmin         = low_habitat_limit;    % Individuals' upper bounds.
       
Set.pop_size     = deParameters.I_NP;                  % Population size
step= up_habitat_limit-low_habitat_limit;
            % 
Set.nEvals_Max   = 50000;
        
        Deepso.memGBestMaxSize = ceil(Set.pop_size * 1 );%antes 0.2
        %Deepso.mutationRate = 0.8;
        Deepso.mutationRate = 0.8;
        %Deepso.communicationProbability = 0.6;
        Deepso.communicationProbability = 0.6;
        Deepso.localSearchProbability = 0.5;
        Deepso.localSearchContinuousDiscrete = 0.15;
    
        Ff.excludeBranchViolations = 1;
        Ff.factor = 1;
        Ff.numCoefFF = 3;
        Ff.avgCoefFF = zeros( 1, Ff.numCoefFF );
        Ff.coefFF = ones( 1, Ff.numCoefFF );
        Ff.numFFEval = 0;

end