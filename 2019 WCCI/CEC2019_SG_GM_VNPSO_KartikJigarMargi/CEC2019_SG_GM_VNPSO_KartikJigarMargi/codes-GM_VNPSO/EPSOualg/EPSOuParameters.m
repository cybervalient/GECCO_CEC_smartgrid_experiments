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


epsouParameters.I_NP =5;          % population in EPSOu
epsouParameters.r = 3;              % Number of replication times
epsouParameters.stop = 7;          % Stopping criteria
epsouParameters.selecWorse = 3;    % Selection parameter (% of the worse solutions)
