%% Proposed ALGORITHM: Gauss Mapped Variable Neighbourhood Particle Swarm Optimization (GM_VNPSO)
%% Developers: 
% Kartik S. Pandya, Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Jigar Sarda, Asst. Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Margi Shah, Independent Researcher, Gujarat, INDIA


%% References:
%The codes of VNS is available on http://sites.ieee.org/psace-mho/2017-smart-grid-operation-problems-competition-panel/.
%The codes of EPSO is available on http://www.gecad.isep.ipp.pt/WCCI2018-SG-COMPETITION/
%The Codes of VNS were modified by Sergio Rivera, srriverar@unal.edu.co,professor at UN.
% The Codes of EPSO were developed by Phillipe Vilaça Gomes
% The codes have been modified by the developers to propose GM_VNPSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Deepso,Ff,Set] = DEEPSO_SETTINGS(low_habitat_limit,up_habitat_limit,deParameters)

        D=size(low_habitat_limit);
        Set.D            = D(1,2);                   % Variables to optimize
                
        Set.Xmax         = up_habitat_limit;    % Individuals' lower bounds.
        Set.Xmin         = low_habitat_limit;    % Individuals' upper bounds.
                
        Set.pop_size     = deParameters.I_NP;                  % Population size
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