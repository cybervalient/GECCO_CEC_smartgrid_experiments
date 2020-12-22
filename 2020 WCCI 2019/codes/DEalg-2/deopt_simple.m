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
FVr_minbound = low_habitat_limit; %��������
FVr_maxbound = up_habitat_limit; %��������
I_itermax    = deParameters.I_itermax; %number of max iterations/gen

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %ѡ�񳬹�������������޸Ľ���ķ���
                               %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc=  otherParameters.fnc; %ѡ��fitness function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----Check input variables---------------------------------------------
if (I_NP < 5) %ÿ������ѡ��5���Ӵ�
   I_NP=5;
   fprintf(1,' I_NP increased to minimal value 5\n');
end
if ((F_CR < 0) || (F_CR > 1))
   F_CR=0.5;
   fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
end
if (I_itermax <= 0) %Ҫ����0��iteration��5000/I_itermax 5000/10=500��
   I_itermax = 200;
   fprintf(1,'I_itermax should be > 0; set to default value 200\n');
end

%-----Initialize population and some arrays-------------------------------
%FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-allocation of loop variables
fitMaxVector = nan(1,I_itermax); %1*500��NaN����
% limit iterations by threshold
gen = 1; %iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----FM_pop is a matrix of size I_NPx(I_D+1)=10*3409. It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1); %����10*3408�ľ���ÿ����ͬ��ÿ��Ϊ��������
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1); %����10*3408�ľ���ÿ����ͬ��ÿ��Ϊ��������
deParameters.minPositionsMatrix=minPositionsMatrix; %����deParameters��
deParameters.maxPositionsMatrix=maxPositionsMatrix; %����deParameters��

% generate initial population.������ɵĵ�һ��10*3408���Ӵ���10Ϊ�Ӵ�������3408Ϊ��������
rand('state',otherParameters.iRuns) %Guarantee same initial population
FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix); %������ɵĵ�һ��10*3408���Ӵ���10Ϊ�Ӵ�������3408Ϊ��������


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
[S_val, ~]=feval(fnc,FM_pop,caseStudyData,otherParameters);%10���Ӵ���fitness functionֵ��10*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S_bestval,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
fitMaxVector(1,gen) = S_bestval;%����ʼ����Ӵ���fitnessֵ����fitMaxVector�У�1*500

%------DE-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------
FVr_rot  = (0:1:I_NP-1);               % rotating index array (size I_NP)
while gen<I_itermax %%&&  fitIterationGap >= threshold I_itermax=500
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %FM_ui��10*3408���¾���ͨ��differential variation���ɡ�
    %FM_base��10*3408���¾��󣬴�����˳���FM_pop��
    [FM_ui,FM_base,~]=generate_trial(I_strategy,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP, I_D, FVr_rot);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

    %% Boundary Control
    %%�ж��²������Ƿ����������������Ҫ�󣬽���������滻��case1�滻�������޵�ֵ��case2�滻��������֮����������ͨ��BRMѡ��case���˴�BRM=2��
    FM_ui=update(FM_ui,minPositionsMatrix,maxPositionsMatrix,BRM,FM_base);
    
   
    %Evaluation of new Pop
    [S_val_temp, ~]=feval(fnc,FM_ui,caseStudyData, otherParameters); %S_val_tempΪ10*1�ľ��󣬴洢��FM_ui��fitnessֵ
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Elitist Selection
    %ѡ�������и��õģ��滻��ԭ���ж�Ӧ��fitnessֵ�ͱ���ֵ
    ind=find(S_val_temp<S_val);%��S_val_temp��10*1������һ��S_val��10*1���Աȣ��ҳ���S_val_temp��С��fitnessֵ
    S_val(ind)=S_val_temp(ind);%��S_val_temp��С��fitnessֵ��S_val�滻
    FM_pop(ind,:)=FM_ui(ind,:);%�滻��Ӧ�ı���ֵ
  
  
    %% update best results
    %���¶�Ӧ10���Ӵ��еĵ����ֵ
    [S_bestval,I_best_index] = min(S_val);
    FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
    % store fitness evolution and obj fun evolution as well
     
    fprintf('Fitness value: %f\n',fitMaxVector(1,gen) )
    fprintf('Generation: %d\n',gen)
 
    gen=gen+1;
    %�����fitness�浽fitMaxVector��
    fitMaxVector(1,gen) = S_bestval;

end %---end while ((I_iter < I_itermax) ...
%����С��fitnessֵ��fitMaxVector�еĵ�500��������Fit_and_p
%���������10������ͬһ���ط�
figure(1)
plot(fitMaxVector,'Linewidth',2)%
grid on
xlabel('��������' ); ylabel('��Ӧ��');
Fit_and_p=[fitMaxVector(1,gen) 0]; %;p2;p3;p4]


 
% VECTORIZED THE CODE INSTEAD OF USING FOR ������ʼ��
function pop=genpop(a,b,lowMatrix,upMatrix) %���ڲ�����ʼ�壬a=10,b=3408
pop=unifrnd(lowMatrix,upMatrix,a,b);

% VECTORIZED THE CODE INSTEAD OF USING FOR
%�ж��²������Ƿ����������������Ҫ�󣬽���������滻��case1�滻�������޵�ֵ��case2�滻��������֮��������
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
  FVr_ind = randperm(4);               % index pointer array 1*4�ľ���������1~4�������
  FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors 1*10�ľ���������1~10�������
  FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions ��0~9������FVr_rot����ǰ����ƶ�1��4��λ��ind(1)�����ڽ�FVr_a1��˳������FVr_a2
  FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations ��FVr_a1˳��FVr_rt+1��λ��
  FVr_rt  = rem(FVr_rot+FVr_ind(2),I_NP);     % rotate indices by ind(2) positions ��0~9������FVr_rot����ǰ����ƶ�1��4��λ��ind(2)�����ڽ�FVr_a2��˳������FVr_a3
  FVr_a3  = FVr_a2(FVr_rt+1);                % rotate vector locations ��FVr_a2˳��FVr_rt+1��λ��
  %�������10�������˳������FM_pm1��FM_pm2��FM_pm3
  FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1
  FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
  FM_pm3 = FM_popold(FVr_a3,:);             % shuffled population 3
  FM_mui = rand(I_NP,I_D) < F_CR;  % all random numbers < F_CR are 1, 0 otherwise ����10*3408��0��1����С��F_CR��0.9��Ϊ1��1�Ƚ϶�
  FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui ��FM_mui��0��1����������0�Ƚ϶�

	switch method
        case 1
            FM_ui = FM_pm3 + F_weight*(FM_pm1 - FM_pm2);   % differential variation �����µ��� FM_ui 10*3408
            FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover ͨ�����FM_ui��FM_popold�����µ��� FM_ui 10*3408����FM_uiΪ��
            FM_base = FM_pm3; %��FM_pm3������˳���FM_popold���浽FM_base��
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

