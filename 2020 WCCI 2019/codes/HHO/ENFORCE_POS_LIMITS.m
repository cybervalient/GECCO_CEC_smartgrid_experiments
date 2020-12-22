%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/


function [ new_pos, new_vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax, pop_size, D )
% Enforces search space limits

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
% from duplicated CBBO
% Make sure there are no duplicate individuals in the population.
% This logic does not make 100% sure that no duplicates exist, but any duplicates that are found are
% randomly mutated, so there should be a good chance that there are no duplicates after this procedure.
for i = 1 : size(new_pos)
    Chrom1 = sort(new_pos(i,:));
    for j = i+1 : size(new_pos)
        Chrom2 = sort(new_pos(j,:));
        if isequal(Chrom1, Chrom2)
            new_pos(j,:) = Xmin + ((Xmax - Xmin) * rand);
        end
    end
end


end