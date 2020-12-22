function [ new_pos, new_vel ] = COMPUTE_NEW_POS( pos, vel )
% Computes new position for the particles and updates its velocity
new_pos = pos + vel;
new_vel = vel;
end