%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:         [FVr_bestmem,S_bestval,I_nfeval] = deopt(fname,S_struct)
% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017

function [Fit_and_p,FVr_bestmemit, fitMaxVector] = ...
    deopt_simple(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = deParameters.I_NP; % Size of the population in DE = 10
F_weight     = deParameters.F_weight; %Mutation factor
F_CR         = deParameters.F_CR; %Recombination constant
I_D          = numel(up_habitat_limit); %Number of variables or dimension
deParameters.nVariables=I_D;
FVr_minbound = low_habitat_limit; %变量下限
FVr_maxbound = up_habitat_limit; %变量上限
I_itermax    = deParameters.I_itermax; %number of max iterations/gen

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %选择超过上下限情况下修改结果的方法
                               %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc=  otherParameters.fnc; %选择fitness function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----Check input variables---------------------------------------------
if (I_NP < 5) %每次至少选择5个子代
   I_NP=5;
   fprintf(1,' I_NP increased to minimal value 5\n');
end
if ((F_CR < 0) || (F_CR > 1))
   F_CR=0.5;
   fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
end
if (I_itermax <= 0) %要大于0次iteration（5000/I_itermax 5000/10=500）
   I_itermax = 200;
   fprintf(1,'I_itermax should be > 0; set to default value 200\n');
end

%-----Initialize population and some arrays-------------------------------
%FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-allocation of loop variables
fitMaxVector = nan(1,I_itermax); %1*500的NaN矩阵
% limit iterations by threshold
gen = 1; %iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----FM_pop is a matrix of size I_NPx(I_D+1)=10*3409. It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1); %产生10*3408的矩阵，每列相同，每行为变量下限
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1); %产生10*3408的矩阵，每列相同，每行为变量上限
deParameters.minPositionsMatrix=minPositionsMatrix; %存入deParameters中
deParameters.maxPositionsMatrix=maxPositionsMatrix; %存入deParameters中

% generate initial population.随机生成的第一代10*3408的子代（10为子代数量，3408为变量数）
rand('state',otherParameters.iRuns) %Guarantee same initial population
FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix); %随机生成的第一代10*3408的子代（10为子代数量，3408为变量数）


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
[S_val, ~]=feval(fnc,FM_pop,caseStudyData,otherParameters);%10个子代的fitness function值，10*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S_bestval,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
fitMaxVector(1,gen) = S_bestval;%将初始最佳子代的fitness值存入fitMaxVector中，1*500

%------DE-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------
FVr_rot  = (0:1:I_NP-1);               % rotating index array (size I_NP)
while gen<I_itermax %%&&  fitIterationGap >= threshold I_itermax=500
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %FM_ui：10*3408的新矩阵，通过differential variation生成。
    %FM_base：10*3408的新矩阵，打乱行顺序的FM_pop。
    [FM_ui,FM_base,~]=generate_trial(I_strategy,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP, I_D, FVr_rot);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

    %% Boundary Control
    %%判断新产的族是否各变量满足上下限要求，将不满足的替换，case1替换成上下限的值，case2替换成上下限之间的随机数（通过BRM选择case，此处BRM=2）
    FM_ui=update(FM_ui,minPositionsMatrix,maxPositionsMatrix,BRM,FM_base);
    
   
    %Evaluation of new Pop
    [S_val_temp, ~]=feval(fnc,FM_ui,caseStudyData, otherParameters); %S_val_temp为10*1的矩阵，存储新FM_ui的fitness值
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Elitist Selection
    %选择新族中更好的，替换掉原族中对应的fitness值和变量值
    ind=find(S_val_temp<S_val);%新S_val_temp（10*1）与上一次S_val（10*1）对比，找出新S_val_temp更小的fitness值
    S_val(ind)=S_val_temp(ind);%将S_val_temp更小的fitness值给S_val替换
    FM_pop(ind,:)=FM_ui(ind,:);%替换对应的变量值
  
  
    %% update best results
    %更新对应10个子代中的的最佳值
    [S_bestval,I_best_index] = min(S_val);
    FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
    % store fitness evolution and obj fun evolution as well
     
    fprintf('Fitness value: %f\n',fitMaxVector(1,gen) )
    fprintf('Generation: %d\n',gen)
 
    gen=gen+1;
    %将最佳fitness存到fitMaxVector中
    fitMaxVector(1,gen) = S_bestval;

end %---end while ((I_iter < I_itermax) ...
%将最小的fitness值（fitMaxVector中的第500个）存入Fit_and_p
%结果分析，10个球到了同一个地方
figure(1)
plot(fitMaxVector,'Linewidth',2)%
grid on
xlabel('进化代数' ); ylabel('适应度');
Fit_and_p=[fitMaxVector(1,gen) 0]; %;p2;p3;p4]


 
% VECTORIZED THE CODE INSTEAD OF USING FOR 创建初始族
function pop=genpop(a,b,lowMatrix,upMatrix) %用于产生初始族，a=10,b=3408
pop=unifrnd(lowMatrix,upMatrix,a,b);

% VECTORIZED THE CODE INSTEAD OF USING FOR
%判断新产的族是否各变量满足上下限要求，将不满足的替换，case1替换成上下限的值，case2替换成上下限之间的随机数
function p=update(p,lowMatrix,upMatrix,BRM,FM_base)
switch BRM
    case 1 %Our method
        %[popsize,dim]=size(p);
        [idx] = find(p<lowMatrix);
        p(idx)=lowMatrix(idx);
        [idx] = find(p>upMatrix);
        p(idx)=upMatrix(idx);
    case 2 %Random reinitialization
        [idx] = [find(p<lowMatrix);find(p>upMatrix)];
        replace=unifrnd(lowMatrix(idx),upMatrix(idx),length(idx),1);
        p(idx)=replace;
    case 3 %Bounce Back
      [idx] = find(p<lowMatrix);
      p(idx)=unifrnd(lowMatrix(idx),FM_base(idx),length(idx),1);
        [idx] = find(p>upMatrix);
      p(idx)=unifrnd(FM_base(idx), upMatrix(idx),length(idx),1);
end

function [FM_ui,FM_base,msg]=generate_trial(method,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP,I_D,FVr_rot)

  FM_popold = FM_pop;                  % save the old population
  FVr_ind = randperm(4);               % index pointer array 1*4的矩阵，里面是1~4的随机数
  FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors 1*10的矩阵，里面是1~10的随机数
  FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions 将0~9的数列FVr_rot，向前随机移动1到4个位置ind(1)，用于将FVr_a1的顺移生成FVr_a2
  FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations 将FVr_a1顺移FVr_rt+1个位置
  FVr_rt  = rem(FVr_rot+FVr_ind(2),I_NP);     % rotate indices by ind(2) positions 将0~9的数列FVr_rot，向前随机移动1到4个位置ind(2)，用于将FVr_a2的顺移生成FVr_a3
  FVr_a3  = FVr_a2(FVr_rt+1);                % rotate vector locations 将FVr_a2顺移FVr_rt+1个位置
  %将输入的10个族打乱顺序，生成FM_pm1，FM_pm2，FM_pm3
  FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1
  FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
  FM_pm3 = FM_popold(FVr_a3,:);             % shuffled population 3
  FM_mui = rand(I_NP,I_D) < F_CR;  % all random numbers < F_CR are 1, 0 otherwise 生成10*3408的0、1矩阵，小于F_CR（0.9）为1，1比较多
  FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui 将FM_mui的0，1，反过来，0比较多

	switch method
        case 1
            FM_ui = FM_pm3 + F_weight*(FM_pm1 - FM_pm2);   % differential variation 生成新的族 FM_ui 10*3408
            FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover 通过组合FM_ui和FM_popold生成新的族 FM_ui 10*3408，以FM_ui为主
            FM_base = FM_pm3; %将FM_pm3（打乱顺序的FM_popold）存到FM_base中
            msg=' DE/rand/bin';
        case 2
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            %VEC by FLC
            FM_bm=repmat(FVr_bestmemit,I_NP,1);
            FM_ui = FM_popold + F_weight*(FM_bm-FM_popold) + F_weight*(FM_pm1 - FM_pm2);
            FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;
            FM_base = FM_bm;
            msg=' DE/current-to-best/1';
	end
return

