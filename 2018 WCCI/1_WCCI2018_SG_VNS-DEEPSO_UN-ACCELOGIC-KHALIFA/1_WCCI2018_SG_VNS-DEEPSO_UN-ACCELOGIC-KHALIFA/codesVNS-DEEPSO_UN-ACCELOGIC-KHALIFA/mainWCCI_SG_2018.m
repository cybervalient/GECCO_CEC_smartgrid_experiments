%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD WCCI2018: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


clear all;clc;close all; tTotalTime=tic; % lets track total computational time
addpath('CallDataBases') 
addpath('DEalg') %Participants should add to the path the folder with the code of their algorithms 
noRuns = 20; % Number of trials here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data base 
caseStudyData=callDatabase(1);% No.Scenarios: (1) 100 scenarios (2) 10 scenarios
                              % For the competition use callDatabase(1)
                              % For making test purposes callDatabase(2)
            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
algorithm='DE_rand'; %'The participants should include their algorithm here'
DEparameters %Function defined by the participant
No_solutions=deParameters.I_NP;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
 otherParameters =setOtherParameters(caseStudyData,No_solutions);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set lower/upper bounds of variables 
[lowerB,upperB] = setVariablesBounds(caseStudyData,otherParameters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some parameters that can be modified by the user
otherParameters.DirectMEthod=2; %1:without direct repair 2:With direct repairs (No violations guarantee)
otherParameters.ensPenalty=100; % Penalty factor:insufficient generation / energy not supplied

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Call the MH for optimizationclear 
            ResDB=struc([]);
         for iRuns=1:noRuns %Number of trails
            tOpt=tic;
            rand('state',sum(noRuns*100*clock))% ensure stochastic indpt trials

            [ResDB(iRuns).Fit_and_p, ...
             ResDB(iRuns).sol, ...
             ResDB(iRuns).fitVector, ...
             ResDB(iRuns).Best_otherInfo] =...
             VNS2(deParameters,caseStudyData,otherParameters,lowerB,upperB);
  
        ResDB(iRuns).tOpt=toc(tOpt); % time of each trial
        
        %size(ResDB(iRuns).fitVector)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Save the results and stats
        Save_results
         end
tTotalTime=toc(tTotalTime); %Total time
%% End of MH Optimization
 