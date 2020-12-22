function output = getObjectiveValue( input,Neval,mode )
if(mode==1)
    output = mean(input) + std(input);
else
    inp_mean = sum(input)/Neval;
    output = inp_mean;
end

end

