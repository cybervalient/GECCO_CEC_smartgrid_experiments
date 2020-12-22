%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ new_pos, new_vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax )
% Enforces search space limits
global D
Evdepso_parameters
pop_size=EVDEPSO_parameters.I_NP;
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
