%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
function [ new_vel ] = EVDEPSO_COMPUTE_NEW_VEL( pos, gbest,   vel, Vmin, Vmax, weights )
% Computes new velocity according to the DEEPSO movement rule
global ps
global proc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INERTIA weight WITH NORMAL DISTRIBUTION
Random_inertia_weight_1=vel*weights( 1 )* normrnd( 0, 1 );
%Random INERTIA WITH uniform DISTRIBUTION 
 Random_inertia_weight_2 =2*(1+rand()+unifrnd(0,1));
%Add old velocity in current position to enhance the new velocity to get  the best posion of particles with less number of iteration and execution time 
% it is the memory of velocity
new_pos_1=(vel+pos);
% Compute cooperation term
cooperationTerm =weights( 4 )*(( gbest* (1+weights( 3 )) - pos )/2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute velocity
new_vel = Random_inertia_weight_1+new_pos_1+Random_inertia_weight_2+ cooperationTerm ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check velocity limits 
new_vel = ( new_vel > Vmax ) .*(Vmin +( Vmax - Vmin )/4)+ ( new_vel <= Vmax ) .* new_vel;
new_vel = ( new_vel < Vmin ) .* (Vmin + ( Vmax - Vmin )/4) + ( new_vel >= Vmin ) .* new_vel;
end