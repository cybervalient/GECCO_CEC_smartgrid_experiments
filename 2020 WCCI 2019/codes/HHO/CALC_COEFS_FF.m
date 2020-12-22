function CALC_COEFS_FF()
% Calculates new coefficients for the fitness function
global ff_par;
maxCoef = max( ff_par.avgCoefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) );
for i = 1 : ff_par.numCoefFF - ff_par.excludeBranchViolations
    if ff_par.avgCoefFF( i ) > 0
        if ff_par.coefFF( i ) == maxCoef
            ff_par.coefFF( i ) = 1;
        else
            ff_par.coefFF( i ) = maxCoef / ff_par.avgCoefFF( i );
        end
    end
end
ff_par.coefFF = round( log10( ff_par.coefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) ) );
for i = 1 : ff_par.numCoefFF - ff_par.excludeBranchViolations
    if ff_par.coefFF( i ) == Inf || ff_par.coefFF( i ) == -Inf
        ff_par.coefFF( i ) = 1;
    end
    ff_par.coefFF( i ) = 10 ^ ff_par.coefFF( i );
end
if ff_par.excludeBranchViolations == 1
    ff_par.coefFF( ff_par.numCoefFF ) = max( ff_par.coefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) );
end
end