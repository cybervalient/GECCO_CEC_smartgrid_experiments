%% TEAM: UN-ACCELOGIC-KHALIFA
% Cooperation of Universidad Nacional de Colombia (UN), ACCELOGIC and Khalifa University
%% TEAM MEMBERS: 
% Sergio Rivera, srriverar@unal.edu.co, professor at UN
% Pedro Garcia, pjgarciag@unal.edu.co, PhD Student at UN and Researcher at Servicio Nacional de Aprendizaje SENA Ocaña, Colombia
% Julian Cantor, jrcantorl@unal.edu.co, Graduated from UN
% Juan Gonzalez, juan.gonzalez@accelogic.com, Chief Scientist at ACCELOGIC
% Rafael Nunez, rafael.nunez@accelogic.com, Vice President Research and Development at ACCELOGIC
% Camilo Cortes, caacortesgu@unal.edu.co, professor at UN
% Alejandra Guzman, maguzmanp@unal.edu.co, professor at UN
% Ameena Al Sumaiti  ameena.alsumaiti@ku.ac.ae, professor at Khalifa University
%% ALGORITMH: VNS-DEEPSO
% Combination of Variable Neighborhood Search algorithm (VNS) and Differential Evolutionary Particle Swarm Optimization (DEEPSO)
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

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