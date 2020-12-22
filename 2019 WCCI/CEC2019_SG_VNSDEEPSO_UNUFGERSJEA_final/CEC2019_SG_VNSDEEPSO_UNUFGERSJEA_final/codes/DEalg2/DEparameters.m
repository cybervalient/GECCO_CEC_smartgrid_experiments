%% TEAM: UN-UF-GERS-JEA
% Cooperation: Universidad Nacional de Colombia, University of Florida, GERS USA and JEA
%% TEAM MEMBERS: 
% Pedro Garcia, pjgarciag@unal.edu.co, PhD Student at UN 
% Diego Rodriguez, diego.rodriguez@gers.com.co, International Studies Manager at GERS USA and PhD Student at UN
% David Alvarez, dlalvareza@unal.edu.co, Postdoc at UN
% Sergio Rivera, srriverar@unal.edu.co, Professor at UN and Fulbright Scholar
% Camilo Cortes, caacortesgu@unal.edu.co, Professor at UN
% Alejandra Guzman, maguzmanp@unal.edu.co, Professor at UN
% Arturo Bretas, arturo@ece.ufl.edu, Professor at UF
% Julio Romero, romeje@jea.com, Chief Innovation and Transformation Officer at JEA
%% ALGORITMH: VNS-DEEPSO
% Combination of Variable Neighborhood Search algorithm (VNS) and Differential Evolutionary Particle Swarm Optimization (DEEPSO)
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2017 and 2018 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems

% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017
deParameters.Scenarios   = 100; % ESTE VALOR PUEDE VARIA DE 0-500


deParameters.I_NP= 1; % population in DE
deParameters.F_weight= 0.3; %Mutation factor
deParameters.F_CR= 0.5; %Recombination constant
deParameters.I_strategy   = 1; %DE strategy
deParameters.I_itermax = (50000 - deParameters.I_NP*deParameters.Scenarios)/(deParameters.I_NP*deParameters.Scenarios);
deParameters.I_iterma = round(deParameters.I_itermax);


deParameters.I_itermax*deParameters.I_NP*deParameters.Scenarios+deParameters.I_NP*deParameters.Scenarios
deParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back

%

