% Author: Yuan SUN
% Email:  yuan.sun@rmit.edu.au 
%         suiyuanpku@gmail.com
%
% ------------
% Description:
% ------------
% RDG3 - This function runs the recursive differential grouping 3
%        procedure to decompose the benchmark overlapping problems
%
% -------
% Inputs:
% -------
%    fun        : the function suite for which the interaction structure 
%                 is going to be identified in this case benchmark
%                 overlapping problem
%
%    fun_number : the function number.
%
%    options    : this variable contains the options such as problem
%                 dimensionality, upper and lower bounds.
%
% --------
% Outputs:W
% --------
%    seps       : a vector of all separable variables.
%    nongroups  : a cell array containing all non-separable groups.
%    FEs        : the total number of fitness evaluations used.


function [seps, nongroups, FEs,bestx] =half_RDG3(caseStudyData,otherParameters, lb,ub,tn)

    bestval = 10000;
    bestx=0;
    fun = otherParameters.fnc;
    dummy = find(lb == ub);
    %dim       = numel(lb)/2;
    seps      = [];
    nongroups = {};
    FEs       = 0; 
    oneday_dim = numel(lb)/24;
    p1  = lb;
    %y1  = feval(fun, p1, caseStudyData, otherParameters);
    y1 = 0;
    xremain = [];
    for t = 1:12
        xremain = [xremain,oneday_dim*2*(t-1)+1:oneday_dim*2*(t-1)+oneday_dim];
    end
    
    if numel(dummy) ~= 0
        for i = 1:numel(dummy)
            if ismember(dummy(i),xremain)
                xremain(find(dummy(i)==xremain)) = [];
            end 
        end
    end
    %xremain = xremain(randperm(length(xremain)));
    dim       = numel(xremain);

    sub1 = xremain(1);
    sub2 = xremain(2:end);
   
    while size(xremain,2) > 0 
        xremain = [];
        [sub1_a,xremain,FEs,temp_bestx,temp_bestval] = half_INTERACT(caseStudyData,otherParameters,sub1,sub2,p1,y1,FEs,xremain,lb,ub,bestx,bestval);
        if temp_bestval < bestval
            bestval = temp_bestval;
            bestx = temp_bestx;
        end
        if length(sub1_a) ~= length(sub1) && length(sub1_a) < tn             
            sub1 = sub1_a;
            sub2 = xremain;
            if size(xremain,2) == 0
                nongroups = {nongroups{1:end}, sub1};
                break;
            end                  
        else 
            if length(sub1_a) == 1
               seps = [seps;sub1_a];
            else
               nongroups = {nongroups{1:end},sub1_a};
            end 
            
            if length(xremain) > 1
                sub1 = xremain(1);
                xremain(1) = [];
                sub2 = xremain(1:end);
            else if length(xremain) == 1
                seps = [seps;xremain(1)];
                break;
            end                       
        end   
    end
    end
    