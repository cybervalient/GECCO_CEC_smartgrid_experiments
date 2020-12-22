function [ mutated_pos ] = LOCAL_SEARCH_CONTINUOUS( pos, Xmin, Xmax )
tmpRnd = normrnd( 0, 1 );
if tmpRnd < 0
    delta = pos - Xmin;
else
    delta = Xmax - pos;
end
mutated_pos = pos + tmpRnd * delta * 0.25;
end