% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

deParameters.I_NP= 10; % population  
deParameters.I_itermax=100; 
%% DE parameters
deParameters.F1= 0.7; %Scalar number  

%% BEE parameters
deParameters.L=10; %limit (bees scout)

deParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back
% valores para la auto mutacion

