%%TEAM: UN-ACCELOGIC-KHALIFA
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
%% 
% THIS SCRIPT IS BASED ON THE CODE SUPPLIED BY THE DEVELPERS OF THIS
% COMPETITION

% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

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

