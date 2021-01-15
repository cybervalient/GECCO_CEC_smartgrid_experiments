%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ALGORITMH: HL_PS_VNS
%HYBRID LEVY PARTICLE SWARM VARIABLE NEIGHBORHOOD SEARCH OPTIMIZATION
%% Developers: 
% Dharmesh A. Dabhi, Assistant Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
% Kartik S. Pandya, Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function [Deepso,Ff,Set] = HL_PS_VNSO_SETTINGS(low_habitat_limit,up_habitat_limit,HL_PS_VNS_Parameters)

        D=size(low_habitat_limit);
        Set.D            = D(1,2);                   % Variables to optimize
                
        Set.Xmax         = up_habitat_limit;    % Individuals' lower bounds.
        Set.Xmin         = low_habitat_limit;    % Individuals' upper bounds.
                
        Set.pop_size     = HL_PS_VNS_Parameters.I_NP;                  % Population size
        Set.nEvals_Max   = 50000;                % 
       
        Deepso.memGBestMaxSize = ceil(Set.pop_size * 1 );%antes 0.2
        %Deepso.mutationRate = 0.8;
        Deepso.mutationRate = 0.8;
        %Deepso.communicationProbability = 0.6;
        Deepso.communicationProbability = 0.6;
        Deepso.localSearchProbability = 0.01;
        Deepso.localSearchContinuousDiscrete = 0.15;
    
        Ff.excludeBranchViolations = 1;
        Ff.factor = 1;
        Ff.numCoefFF = 3;
        Ff.avgCoefFF = zeros( 1, Ff.numCoefFF );
        Ff.coefFF = ones( 1, Ff.numCoefFF );
        Ff.numFFEval = 0;
    
end