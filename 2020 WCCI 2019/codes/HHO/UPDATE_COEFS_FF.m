% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function UPDATE_COEFS_FF( tmpABC )
% Adds new observations to the memory used to update the coefficients of the fitness function
global ff_par;
if ff_par.numFFEval == 1
    ff_par.avgCoefFF = tmpABC;
else
    for i = 1 : ff_par.numCoefFF
        ff_par.avgCoefFF( i ) = ff_par.avgCoefFF( i ) * ( ( ff_par.numFFEval  - 1 ) / ff_par.numFFEval ) + tmpABC( i ) / ff_par.numFFEval;
    end
end
end