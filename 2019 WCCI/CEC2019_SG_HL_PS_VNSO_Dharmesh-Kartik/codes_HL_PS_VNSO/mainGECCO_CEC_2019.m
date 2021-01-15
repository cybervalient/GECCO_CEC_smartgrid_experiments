
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ALGORITMH: HL_PS_VNSO
%HYBRID LEVY PARTICLE SWARM VARIABLE NEIGHBORHOOD SEARCH OPTIMIZATION
%% Developers: 
% Dharmesh A. Dabhi, Assistant Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
% Kartik S. Pandya, Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
%% References:
%The codes of VNS available on http://sites.ieee.org/psace-mho/2017-smart-grid-operation-problems-competition-panel/. 
% The Codes of DEEPSO were developed by V. Miranda, Associate Director,IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO),INESC-TEC, Portugal  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/ 
% The codes have been modified by the developers to propose HL_PS_VNS

clear all;clc;close all; 
tTotalTime=tic; % lets track total computational time
addpath('CallDataBases','Functions') 

Select_Algorithm=1;
%1: DE algorithm (test algorithm)
%2: Your algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
 switch Select_Algorithm
     case 1
        addpath('DEalg')
        algorithm='HL_PS_VNSO'; %'The participants should include their algorithm here'
        VNS_Parameters %Function defined by the participant
No_solutions=HL_PS_VNS_Parameters.I_NP;
     case 2
%         
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
 otherParameters.No_eval_Scenarios=500;
%    otherParameters.otherParametersone.No_eval_Scenarios=10; %Requiered for single evaluation
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
% otherParameters.ensPenalty=1e2; % Penalty factor:insufficient generation / energy not supplied
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
                    HL_PS_VNSO(HL_PS_VNS_Parameters,caseStudyData,otherParameters,lowerB,upperB);
                case 2
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
 