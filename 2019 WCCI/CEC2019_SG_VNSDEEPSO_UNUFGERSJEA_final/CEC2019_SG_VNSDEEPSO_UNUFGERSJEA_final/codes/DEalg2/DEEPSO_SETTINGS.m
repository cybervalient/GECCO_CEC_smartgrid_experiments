%% TEAM: UN-UF-GERS-JEA
% Cooperation: Universidad Nacional de Colombia, University of Florida, GERS USA and JEA
%% TEAM MEMBERS: 
% Pedro Garcia, pjgarciag@unal.edu.co, PhD Student at UN 
% Diego Rodriguez, diego.rodriguez@gers.com.co, International Studies Manager at GERS USA and PhD Student at UN
% David Alvarez, dlalvareza@unal.edu.co, Postdoc at UN
% Sergio Rivera, srriverar@unal.edu.co, Professor at UN and Fulbright Scholar
% Camilo Cortes, caacortesgu@unal.edu.co, Professor at UN
% Alejandra Guzman, maguzmanp@unal.edu.co, Professor at UN
% Arturo Bretas, arturo@ece.ufl.edu, Professor at UF
% Julio Romero, romeje@jea.com, Chief Innovation and Transformation Officer at JEA
%% ALGORITMH: VNS-DEEPSO
% Combination of Variable Neighborhood Search algorithm (VNS) and Differential Evolutionary Particle Swarm Optimization (DEEPSO)
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2017 and 2018 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems

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