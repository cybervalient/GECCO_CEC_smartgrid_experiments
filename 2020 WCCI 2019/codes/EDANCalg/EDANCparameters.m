% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017
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

deParameters.I_NP= 14; % population in DE 10
deParameters.F_weight= 0.3; %Mutation factor
deParameters.F_CR= 0.5; %Recombination constant
deParameters.I_itermax= 254; % number of max iterations/gen 499
deParameters.I_strategy   = 1; %DE strategy

deParameters.I_bnd_constr = 1; %Using bound constraints 
% 1 repair to the lower or upper violated bound 
% 2 rand value in the allowed range
% 3 bounce back


