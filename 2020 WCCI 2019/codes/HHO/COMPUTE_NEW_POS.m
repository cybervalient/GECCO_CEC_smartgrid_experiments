%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function [ new_pos, new_vel ] = COMPUTE_NEW_POS( pos, vel )
% Computes new position for the particles and updates its velocity
new_pos = pos + vel;
new_vel = vel;
end