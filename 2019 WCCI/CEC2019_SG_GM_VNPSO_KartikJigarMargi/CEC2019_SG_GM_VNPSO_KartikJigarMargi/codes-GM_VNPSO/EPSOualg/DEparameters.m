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

deParameters.I_NP= 1; % population in DE %PARA VNS ESTE VALOR SOLO PUEDE SER 1
deParameters.F_weight= 0.3; %Mutation factor
deParameters.F_CR= 0.5; %Recombination constant

deParameters.Scenarios   = 500; % ESTE VALOR PUEDE VARIA DE 0-100

deParameters.I_itermax = (50000 - deParameters.I_NP*deParameters.Scenarios)/(deParameters.I_NP*deParameters.Scenarios);
deParameters.I_iterma = round(deParameters.I_itermax);

%deParameters.I_itermax= 499; %499; % number of max iterations/gen
deParameters.I_strategy   = 1; %DE strategy


deParameters.I_itermax*deParameters.I_NP*deParameters.Scenarios+deParameters.I_NP*deParameters.Scenarios

deParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back

