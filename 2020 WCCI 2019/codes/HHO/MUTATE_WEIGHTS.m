%  THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/


function [ mutated_Weights ] = MUTATE_WEIGHTS( weights, mutationRate )
% Mutate weights & check weights limits
mutated_Weights = weights;
for i = 1 : 4
    mutated_Weights( i ) = weights( i ) + normrnd( 0, 1 ) * mutationRate;
    if  mutated_Weights( i ) > 1
        mutated_Weights( i ) = 1;
    elseif  mutated_Weights( i ) < 0
        mutated_Weights( i ) = 0;
    end
end
end