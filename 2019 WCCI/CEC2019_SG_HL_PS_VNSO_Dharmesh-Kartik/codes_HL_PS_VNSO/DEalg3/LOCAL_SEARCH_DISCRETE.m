%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ALGORITMH: HL_PS_VNS
%HYBRID LEVY PARTICLE SWARM VARIABLE NEIGHBORHOOD SEARCH OPTIMIZATION
%% Developers: 
% Dharmesh A. Dabhi, Assistant Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
% Kartik S. Pandya, Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/


function [ mutated_pos ] = LOCAL_SEARCH_DISCRETE( pos, Xmin, Xmax )
tmpPos = rounding( pos );
minPos = rounding( Xmin );
maxPos = rounding( Xmax );
if rand() > 0.5
    if tmpPos < maxPos
        mutated_pos = tmpPos + 1;
    else
        mutated_pos = tmpPos - 1;
    end
else
    if tmpPos > minPos
        mutated_pos = tmpPos - 1;
    else
        mutated_pos = tmpPos + 1;
    end
end
end