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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% THIS SCRIPT IS BASED ON THE CODE SUPPLIED BY THE DEVELPERS OF THIS
% COMPETITION

% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

HL_PS_VNS_Parameters.I_NP= 5; % population in DE
% HL_PS_VNS_Parameters.F_weight= 0.2; %Mutation factor
% HL_PS_VNS_Parameters.F_CR= 0.3; %Recombination constant

HL_PS_VNS_Parameters.Scenarios   = 500;

HL_PS_VNS_Parameters.I_itermax = (50000 - HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios)/(HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios);
HL_PS_VNS_Parameters.I_iterma = round(HL_PS_VNS_Parameters.I_itermax)

%deParameters.I_itermax= 499; %499; % number of max iterations/gen
HL_PS_VNS_Parameters.I_strategy   = 1; %DE strategy


HL_PS_VNS_Parameters.I_itermax*HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios+HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios

HL_PS_VNS_Parameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back


