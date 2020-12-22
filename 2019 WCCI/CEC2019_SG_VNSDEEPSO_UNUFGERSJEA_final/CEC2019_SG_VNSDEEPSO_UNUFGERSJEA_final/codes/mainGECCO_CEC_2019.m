%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;clc;close all; 
tTotalTime=tic; % lets track total computational time
addpath('CallDataBases','Functions') 

Select_Algorithm=2;
%1: DE algorithm (test algorithm)
%2: Your algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
 switch Select_Algorithm
     case 1
        addpath('DEalg')
        algorithm='DE'; %'The participants should include their algorithm here'
        DEparameters %Function defined by the participant
        No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
     case 2
        addpath('DEalg2')
        algorithm='DE3'; %'The participants should include their algorithm here'
        DEparameters %Function defined by the participant
        No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
     otherwise
         fprintf(1,' No parameters loaded\n');
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data base 
DB=1; %1 (500) and 2 (1); %Select the database you want to analyze
% 1: CEC2019 500 scenarios
% 2: CEC2019 average scenario
[caseStudyData, DB_name]=callDatabase(DB);

%% Label of the algorithm and the case study
Tag.algorithm=algorithm;
Tag.DB=DB_name;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
otherParameters =setOtherParameters(caseStudyData,No_solutions);
otherParameters.otherParametersone =setOtherParameters(caseStudyData,1); %This is needed to evaluate one solution

 %% Number of scenarios that will be evaluated (this can be changed internally in every algorithm)
 otherParameters.No_eval_Scenarios=100;
 otherParameters.otherParametersone.No_eval_Scenarios=100; %Requiered for single evaluation
 %'Number of scenarios to evaluate cannot be less than 1.'
 
% Participants can simply configure a new value before evaluating the objective function. For example:
% otherParameters.No_eval_Scenarios=20; %Then 20 scenarios will be evaluated 
%[solFitness_M_temp, solPenalties_M_temp,Struct_Eval_temp]=feval(fnc,FM_ui,caseStudyData, otherParameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set lower/upper bounds of variables 
[lowerB,upperB] = setVariablesBounds(caseStudyData,otherParameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Some parameters that can be modified by the user
otherParameters.DirectMEthod=2; %1:without direct repair 2:With direct repairs (No violations guarantee)
otherParameters.ensPenalty=1e2; % Penalty factor:insufficient generation / energy not supplied
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Call the MH for optimization 
noRuns=20; %Set this value to 20 for competition results
ResDB=struc([]);
    for iRuns=1:noRuns %Number of trails
        tOpt=tic;
        rand('state',sum(noRuns*100*clock))% ensure stochastic indpt trials
     
            switch Select_Algorithm
                case 1
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =...
                    deopt_simple(deParameters,caseStudyData,otherParameters,lowerB,upperB);
                case 2
                   [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =...
                    VNS2(deParameters,caseStudyData,otherParameters,lowerB,upperB);
            end 
          
        ResDB(iRuns).tOpt=toc(tOpt); % time of each trial
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Save the results and stats
        Save_results
    end
tTotalTime=toc(tTotalTime); %Total time
%% End of MH Optimization
 