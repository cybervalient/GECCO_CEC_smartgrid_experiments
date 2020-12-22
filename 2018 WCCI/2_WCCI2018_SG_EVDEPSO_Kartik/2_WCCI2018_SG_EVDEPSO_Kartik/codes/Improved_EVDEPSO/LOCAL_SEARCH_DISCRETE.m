%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ mutated_pos ] = LOCAL_SEARCH_DISCRETE( pos, Xmin, Xmax )
tmpPos = round( pos );
minPos = round( Xmin );
maxPos =round( Xmax );
if rand() > 0.4
    if tmpPos < maxPos
        mutated_pos = (tmpPos + 1);
    else
        mutated_pos = (tmpPos - 2);
    end
else
    if tmpPos > minPos
        mutated_pos = (tmpPos - 1);
    else
        mutated_pos = (tmpPos + 2);
    end
end
end