%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ new_pos ] = LOCAL_SEARCH( pos, Xmin, Xmax )
% Mutates the integer part of the particle
global proc
global ps
global EVDEPSO_par
new_pos = pos;
% 
         localSearchContinuousDiscrete = 0.65;
         prob = localSearchContinuousDiscrete;
        if rand() > prob
            %prob = 1 / ps.D_cont;
            prob = 1 / 168;% first active power continuous variables, prob=0.0000208
           for j=0:23
             for i=1+(j*142):7+(j*142);% 1-7, 143-149, 285-291  continuous var
                tmpDim = i;
                if rand() > prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
             end
           end
           
           
           for j=0:23
             for i=15+(j*142):142+(j*142);% remaining continuous variables.15-142, 157-284, 299-426
               
             tmpDim = i;
                if rand() > prob
                 new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
             end
           end
        else
            prob = 1 / 168; % no of discrete variables,0.0060
           
              for j=0:23
                  for i=8+(j*142):14+(j*142) % discrete 8-14, 150: 156, 292:298 generator binaries connected
                 
                   tmpDim=i;
                if rand() > prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_DISCRETE( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
                  end
              end
end
end

