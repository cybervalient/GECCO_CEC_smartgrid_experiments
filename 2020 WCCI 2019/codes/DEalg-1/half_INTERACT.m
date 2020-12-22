% Author: Yuan SUN
% Email: yuan.sun@rmit.edu.au 
%        suiyuanpku@gmail.com
%
% ------------
% Description:
% ------------
% INTERACT - This fncction is used to identify the 
% interaction between two sets of decision variables.
% feval(fnc,x_prime,caseStudyData, otherParameters);

% Author: Yuan SUN
% Email: yuan.sun@rmit.edu.au 
%        suiyuanpku@gmail.com
%
% ------------
% Description:
% ------------
% INTERACT - This function is used to identify the 
% interaction between two sets of decision variables.
%feval(fun,x_prime,caseStudyData, otherParameters);
function [sub1,xremain,FEs,bestx,bestval]= half_INTERACT(caseStudyData,otherParameters,sub1,sub2,p1,y1,FEs,xremain,lb,ub,bestx,bestval)
   %T = randsample(24,1);
   
   %idx = (T-1)*142;
   
   dim = numel(lb);
   
   fun = otherParameters.fnc;
   all_p = zeros(4,dim);
   p2 = p1;
   p2(sub1) = ub(sub1);
   %y2 = feval(fun, p2, caseStudyData, otherParameters);
   %FEs = FEs+1;
   
   p3 = p1;
   p4 = p2;
   %p3(sub2) = (ub+lb)/2.*ones(1,length(sub2));
   %p4(sub2) = (ub+lb)/2.*ones(1,length(sub2));
   %idx
   %sub2
   %size(p3(idx+sub2))
   %size((ub(idx+sub2)+lb(idx+sub2))/2)
   p3(sub2) = (ub(sub2)+lb(sub2))/2;%.*ones(1,length(sub2));
   p4(sub2) = (ub(sub2)+lb(sub2))/2;%.*ones(1,length(sub2));
   
   %y2 = feval(fun, p2, caseStudyData, otherParameters);
   %y3 = feval(fun, p3, caseStudyData, otherParameters);
   %y4 = feval(fun, p4, caseStudyData, otherParameters);
   all_p(1,:) =p1;
   all_p(2,:) =p2;
   all_p(3,:) =p3;
   all_p(4,:) =p4;
   %size(all_p)
   [tt,~]=feval(fun, all_p, caseStudyData, otherParameters);
   [temp_bestval,minidx]=min(tt);
   temp_bestx = all_p(minidx,:);
   if temp_bestval<bestval
       bestval = temp_bestval;
       bestx = temp_bestx;
   end
   
   y1=tt(1);
   y2=tt(2);
   y3=tt(3);
   y4= tt(4);
   delta1 = y1-y2;
   delta2 = y3-y4;
   FEs = FEs+4;

   muM = eps/2;
   gamma = @(n)((n.*muM)./(1-n.*muM));
   epsilon = gamma(dim^0.5+2)*(abs(y1)+abs(y2)+abs(y3)+abs(y4));
  
   if abs(delta1 - delta2) > epsilon
       if length(sub2) == 1
           sub1 = union(sub1,sub2);
       else
           k = floor(length(sub2)/2);
           sub2_1 = sub2(1:k);
           sub2_2 = sub2(k+1:end);       
           [sub1_1,xremain,FEs,bestx,bestval] = half_INTERACT(caseStudyData,otherParameters,sub1,sub2_1,p1,y1,FEs,xremain,lb,ub,bestx,bestval);
           [sub1_2,xremain,FEs,bestx,bestval] = half_INTERACT(caseStudyData,otherParameters,sub1,sub2_2,p1,y1,FEs,xremain,lb,ub,bestx,bestval);
           sub1=union(sub1_1,sub1_2);
       end
   else
       xremain =[xremain,sub2];
   end