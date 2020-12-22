%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ mutated_pos ] = LOCAL_SEARCH_CONTINUOUS( pos, Xmin, Xmax )
tmpRnd = normrnd( 0, 1 );
if tmpRnd < 0
    delta = (pos - Xmin)/4;
else
    delta = (Xmax - pos)/4;
end
mutated_pos = pos + tmpRnd * delta * 0.15;
end


