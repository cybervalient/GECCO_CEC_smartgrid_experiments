function [ new_pos, new_vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax )
% Enforces search space limits
%global proc
%global ps
global D
CHAOS_DEEPSO_parameters
pop_size=chaos_DEEPSO_parameters.I_NP;
new_pos = pos;
new_vel = vel;
for i = 1 : pop_size
    ccrand=rand(1,D);
    ccpos=((1./ccrand)-floor(1./ccrand)); %1*31
    for j = 1 : D
        if new_pos( i, j ) < Xmin( j )
            new_pos( i, j ) = Xmin( j );
           % new_pos( i, j ) = Xmin(j) + ( Xmax(j) - Xmin(j) ).*ccpos( j );
            if new_vel( i, j ) < 0
                new_vel( i, j ) = -new_vel( i, j );
               % new_vel( i, j ) =  -( Vmin(j) + ( Vmax(j) - Vmin(j) ).*ccpos(j));
            end
        elseif new_pos( i, j ) > Xmax( j )
            new_pos( i, j ) = Xmax( j );
           % new_pos( i, j ) =Xmin(j) + ( Xmax(j) - Xmin(j) ).*ccpos( j ) ;
            if new_vel( i, j ) > 0
                new_vel( i, j ) = -new_vel( j );
                % new_vel( i, j ) =  -( Vmin(j) + ( Vmax(j) - Vmin(j) ).*ccpos(j));
            end
        end
        % Check velocity in case of asymmetric velocity limits
        if new_vel( i, j ) < Vmin( j )
            new_vel( i, j ) = Vmin( j );
          %  new_vel( i, j )=( Vmin(j) + ( Vmax(j) - Vmin(j) ).*ccpos(j));
        elseif new_vel( i, j ) > Vmax( j )
            new_vel( i, j ) = Vmax( j );
           % new_vel( i, j )=( Vmin(j) + ( Vmax(j) - Vmin(j) ).*ccpos(j));
        end
    end
end
end