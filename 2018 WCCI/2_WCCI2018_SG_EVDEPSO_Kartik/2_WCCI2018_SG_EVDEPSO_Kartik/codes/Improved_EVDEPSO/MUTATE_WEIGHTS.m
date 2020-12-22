%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ mutated_Weights ] = MUTATE_WEIGHTS( weights, mutationRate )
% Mutate weights & check weights limits
mutated_Weights = weights;
for i  = 1 : 4
    mutated_Weights( i ) = weights( i ) + normrnd( 0, 1 ) * mutationRate;
    if  mutated_Weights( i ) > 1
        mutated_Weights( i ) = 1;%0.6
    elseif  mutated_Weights( i ) < 0
        mutated_Weights( i ) = 0;
    end
end
end