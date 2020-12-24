%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%
% TEAM UC/CISECE-UT3/UCLV
% CUMDANCauchy-C1: a Cellular EDA Designed to Solve the Energy Resource Management Problem Under Uncertainty
%
%
% Phd Student Yoan Martinez Lopez, yoan.martinez@reduc.edu.cu [1,3]
% Phd Ansel Y. Rodriguez-Gonzalez, ansel@cicese.mx [2]
% Phd Julio Madera, julio.madera@reduc.edu.cu [1]
% BSc Student Alexis Moya, alextkmoya@gmail.com [1]
% BSc Student Bismay Morgado Perez, bismaymp@gmail.com [1]
% BSc Student Miguel Betancourt Mayedo, miguel.betancourt@reduc.edu.cu [1]
%
% [1] UC (Universidad de Camaguey, Cuba)
% [2] CISECE-UT3 (Unidad de Transferencia Tecnológica Tepic del Centro de Investigación Científica y de Educación Superior de Ensenada, Mexico)
% [3] UCLV (Universidad de Central de las Villas, Cuba)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;clc;close all; 
tTotalTime=tic; % lets track total computational time
addpath('CallDataBases','Functions') 

Select_Algorithm=10;
%4: edaNC
%5 ABC_DE
%6 CE_CMAES
%7 DEalg-1
%8 GAPSO
%9 AjSO
%10 HFEABC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
 switch Select_Algorithm
     case 4
        addpath('EDANCalg')
        algorithm='DE'; %'The participants should include their algorithm here'
        EDANCparameters %Function defined by the participant
        No_solutions=deParameters.I_NP;   
      case 5
        addpath('ABC_DE')
        algorithm='ABC_DE'; %'The participants should include their algorithm here'
        ABC_parameters %Function defined by the participant
        No_solutions=deParameters.I_NP;   
       case 6
        addpath('path_CE_CMAES')
        algorithm='CE_CMAES'; %'The participants should include their algorithm here'
        all_parameters %Function defined by the participant
        No_solutions=CE_CMAES_parameters.I_NP; 
        case 7
        addpath('DEalg-1')
        algorithm='DEEPSO'; %'The participants should include their algorithm here'
        DEparameters2 %Function defined by the participant
        DEparameters
        No_solutions=deParameters.deepso_NP;
       case 8
        addpath('DEalg-2')
        algorithm='GAPSO'; %'The participants should include their algorithm here'
        DEparameters %Function defined by the participant
        No_solutions=deParameters.I_NP;
       case 9
        addpath('AjSO')
        algorithm='AjSO'; %'The participants should include their algorithm here'
        AjSOparameters %Function defined by the participant
        No_solutions=deParameters.I_NP; 
       case 10
        addpath('HFEABC')
        algorithm='HFEABC'; %'The participants should include their algorithm here'
        HFEABCparameters %Function defined by the participant
        No_solutions=hfeabcParameters.FoodNumber; 
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %Participants can include their algorithms here
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
 otherParameters.No_eval_Scenarios=10;
 otherParameters.otherParametersone.No_eval_Scenarios=10; %Requiered for single evaluation
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
                case 4
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    edaNC(deParameters,caseStudyData,otherParameters,lowerB,upperB);
                  case 5
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    ABC__DE(deParameters,caseStudyData,otherParameters,lowerB,upperB);  
                case 6
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    CE_CMAES(CE_CMAES_parameters,caseStudyData,otherParameters,lowerB,upperB); 
                 case 7
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    doubledeepso(deParameters,caseStudyData,otherParameters,lowerB,upperB);
                case 8
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    GASAPSO(deParameters,caseStudyData,otherParameters,lowerB,upperB);
                case 9
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                    ajso(deParameters,caseStudyData,otherParameters,lowerB,upperB);
                case 10
                    [ResDB(iRuns).Fit_and_p, ...
                    ResDB(iRuns).sol, ...
                    ResDB(iRuns).fitVector, ...
                    ResDB(iRuns).Best_otherInfo] =... 
                   runHFEABC(hfeabcParameters,caseStudyData,otherParameters,lowerB,upperB);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %Your algorithm can be put here
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end 
          
        ResDB(iRuns).tOpt=toc(tOpt); % time of each trial
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Save the results and stats
        Save_results
    end
tTotalTime=toc(tTotalTime); %Total time
%% End of MH Optimization
 