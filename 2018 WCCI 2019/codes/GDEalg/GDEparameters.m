% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

gdeParameters.I_NP= 10; % population in DE
gdeParameters.F_weight= 0.2918; %Mutation factor
gdeParameters.F_CR= 0.4956; %Recombination constant
gdeParameters.I_itermax= 499; % number of max iterations/gen
gdeParameters.I_strategy   = 1; %DE strategy

gdeParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back

gdeParameters.R_rate = 0.0099;
load('obj_par.mat', 'par', 'b')
gdeParameters.R_par = par;
gdeParameters.R_dir = -par / norm(par);

gdeParameters.U_rate = 0.0101;
gdeParameters.U_div = 0.0102;

