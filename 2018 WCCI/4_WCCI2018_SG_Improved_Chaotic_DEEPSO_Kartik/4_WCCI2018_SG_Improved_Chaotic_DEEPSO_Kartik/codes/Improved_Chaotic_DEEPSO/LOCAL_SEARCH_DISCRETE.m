function [ mutated_pos ] = LOCAL_SEARCH_DISCRETE( pos, Xmin, Xmax )
tmpPos = round( pos );
minPos = round( Xmin );
maxPos =round( Xmax );
if rand() > 0.5
    if tmpPos < maxPos
        mutated_pos = tmpPos + 1;
    else
        mutated_pos = tmpPos - 1;
    end
else
    if tmpPos > minPos
        mutated_pos = tmpPos - 1;
    else
        mutated_pos = tmpPos + 1;
    end
end
end