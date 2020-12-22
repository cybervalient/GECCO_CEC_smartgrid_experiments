clear all;clc;close all;
ronds=1;
c1min=[0.3:0.1:2];c1max=[0.3:0.1:2];c2min=[0.3:0.1:2];c2max=[0.3:0.1:2];
for i1=1:numel(c1min)
    for j=1:numel(c1max)
        for k=1:numel(c2min) 
            for l=1:numel(c2max)
clear('caseStudyData','lowerB','No_solutions','noRuns','otherParameters','ResDB','tOpt','tTotalTime','upperB','PSOparameters');
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD WCCI2018: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tTotalTime=tic; % lets track total computational time
addpath('CallDataBases') 
%addpath('DEalg') %Participants should add to the path the folder with the code of their algorithms 
noRuns = 1; % Number of trials here
addpath('Particle_Swarm_Optimization');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data base 
caseStudyData=callDatabase(1);% No.Scenarios: (1) 100 scenarios (2) 10 scenarios
                              % For the competition use callDatabase(1)
                              % For making test purposes callDatabase(2)
            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
%algorithm='DE_rand'; %'The participants should include their algorithm here'
algorithm='PSO';
%DEparameters %Function defined by the participant
psoParameters
PSOparameters.c1min=c1min(i1);
PSOparameters.c1max=c1max(j);
PSOparameters.c2min=c2min(k);
PSOparameters.c2max=c2max(l);
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
        %Save_results
    end
tTotalTime=toc(tTotalTime); %Total time
            results(ronds,:)=[ResDB.Fit_and_p(1,1),c1min(i1),c1max(j),c2min(k),c2max(l)]
            ronds=ronds+1;
            end
        end
    end
end
%% End of MH Optimization
 