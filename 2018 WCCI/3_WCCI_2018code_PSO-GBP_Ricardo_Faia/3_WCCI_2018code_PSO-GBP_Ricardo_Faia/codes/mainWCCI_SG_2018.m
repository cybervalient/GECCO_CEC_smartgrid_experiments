%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD WCCI2018: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;clc;close all; tTotalTime=tic; % lets track total computational time
addpath('CallDataBases') 
%addpath('DEalg') %Participants should add to the path the folder with the code of their algorithms 
noRuns = 20; % Number of trials here
addpath('Particle_Swarm_Optimization');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data base 
caseStudyData=callDatabase(1);% No.Scenarios: (1) 100 scenarios (2) 10 scenarios
                              % For the competition use callDatabase(1)
                              % For making test purposes callDatabase(2)
            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
%algorithm='DE_rand'; %'The participants should include their algorithm here'
algorithm='PSO with Global Best Perturbation - PSO-GBP';
%DEparameters %Function defined by the participant
psoParameters
%No_solutions=deParameters.I_NP;
No_solutions=PSOparameters.nPop;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
 otherParameters =setOtherParameters(caseStudyData,No_solutions);
 otherParameters.um=setOtherParameters(caseStudyData,1);

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set lower/upper bounds of variables 
[lowerB,upperB] = setVariablesBounds(caseStudyData,otherParameters);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some parameters that can be modified by the user
otherParameters.DirectMEthod=2; %1:without direct repair 2:With direct repairs (No violations guarantee)
otherParameters.ensPenalty=100; % Penalty factor:insufficient generation / energy not supplied
otherParameters.um.DirectMEthod=2;
otherParameters.um.ensPenalty=100;


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
             pso_competition(PSOparameters,caseStudyData,otherParameters,lowerB,upperB);
        ResDB(iRuns).tOpt=toc(tOpt); % time of each trial
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Save the results and stats
        Save_results
    end
tTotalTime=toc(tTotalTime); %Total time
%% End of MH Optimization
 