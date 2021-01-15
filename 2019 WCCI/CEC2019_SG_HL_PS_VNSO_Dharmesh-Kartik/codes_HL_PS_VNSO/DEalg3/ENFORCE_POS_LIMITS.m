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


function [ new_pos, new_vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax, pop_size, D )
% Enforces search space limits
%global proc
%global ps
new_pos = pos;
new_vel = vel;
for i = 1 : pop_size
    for j = 1 : D
        if new_pos( i, j ) < Xmin( j )
            new_pos( i, j ) = Xmin( j );
            if new_vel( i, j ) < 0
                new_vel( i, j ) = -new_vel( i, j );
            end
        elseif new_pos( i, j ) > Xmax( j )
            new_pos( i, j ) = Xmax( j );
            if new_vel( i, j ) > 0
                new_vel( i, j ) = -new_vel( j );
            end
        end
        % Check velocity in case of asymmetric velocity limits
        if new_vel( i, j ) < Vmin( j )
            new_vel( i, j ) = Vmin( j );
        elseif new_vel( i, j ) > Vmax( j )
            new_vel( i, j ) = Vmax( j );
        end
    end
end
end