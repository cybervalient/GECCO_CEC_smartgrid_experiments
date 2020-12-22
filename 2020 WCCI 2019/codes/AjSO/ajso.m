%% AUTHOR
% Fabricio Loor, faloor@unsl.edu.ar, PhD Student at UNSL
%% ALGORITMH: AJSO
% This algorithm is based on SHADE. AJSO introduces several changes in
% mutations and in the way of updating the factors of mutation and crosover
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2017 and 2018 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems

%%%%%%%%%%%%%%%%%%%
%% This package is a MATLAB/Octave source code of SHADE 1.1.
%% About SHADE 1.1, please see following papers:
%%
%% * Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.
%%%%%%%%%%%%%%%%%% 

function [Fit_and_p,FVr_bestmemit, fitMaxVector] = ajso(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

I_NP         = deParameters.I_NP;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
deParameters.nVariables = I_D;
I_CR = deParameters.I_CR;
I_F = deParameters.I_F;
FVr_minbound = low_habitat_limit;
FVr_maxbound = up_habitat_limit;
I_evalmax    = deParameters.I_evalmax;
p_best_rate = deParameters.g_p_best_rate;
fnc=  otherParameters.fnc;


minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
ch = 0.5;
check = 1;
gen = 2; %iterations

%%  parameter settings for SHADE
memory_size = I_D;

%% Initialize the main population
FM_popold = unifrnd(minPositionsMatrix,maxPositionsMatrix,I_NP,I_D);
pop = FM_popold ; % the old population becomes the current population
[fitness, ~] = feval(fnc,pop,caseStudyData,otherParameters);
[S_bestval,I_best_index] = min(fitness); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = pop(I_best_index,:); % best member of current iteration
bsf_solution =  pop(I_best_index,:);
bsf_fit_var= S_bestval;
fitMaxVector(1,gen) = S_bestval;
nfes = I_NP;
%%Merories For CR's and F's
memory_sf = normrnd(I_F,0.1,[memory_size,1]);
memory_cr = normrnd(I_CR,0.05,[memory_size,1]);
memory_pos = 1;

strategy = 0;
%%Archive
archive.NP = I_NP; % the maximum size of the archive
archive.pop = pop; % the solutions stored in te archive
archive.funvalues = fitness; % the function value of the archived solutions

FVr_rot  = (0:1:I_NP-1); 
num_reset = 1;
solMinVector = zeros(I_evalmax/I_NP,I_D);

%% main loop
while nfes + I_NP < I_evalmax

	[ ~, sorted_index] = sort(fitness, 'ascend');

	mem_rand_index = ceil(memory_size * rand(I_NP, 1));
	mu_sf = memory_sf(mem_rand_index);
	mu_cr = memory_cr(mem_rand_index);

	%% for generating crossover rate
	cr = normrnd(mu_cr, 0.1);
	%% term_pos = find(mu_cr == -1);
	cr(mu_cr == -1) = 0;
	cr = min(cr, 0.99);
	cr = max(cr, 0.01);

	%% for generating scaling factor
    sf = normrnd(mu_cr, 0.2);
    pos = find(sf <= 0);

    while ~ isempty(pos)
        sf(pos)= mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end


    if(nfes < I_evalmax * 0.5)
        sf = min(sf, ch + 0.1); %%limit the mutation factor value to a maximum
    else
        sf = min(sf, ch + 0.4); %%limit the mutation factor value to a maximum
    end
    jF =  ch * sf;

    %%%%%%%%%%%%%%%%%%%
    FM_popold = pop;                  % save the old population
    FVr_ind = randperm(4);               % index pointer array
    FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors
    FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions
    FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations
    FVr_a3  = randperm(I_NP);
    FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);
    FVr_a4  = FVr_a3(FVr_rt+1);
    FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1
    FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
    FM_pm3 = archive.pop(FVr_a3,:);             % shuffled archived 3
    FM_pm4 = archive.pop(FVr_a4,:);             % shuffled archived 4
    FM_mui = rand(I_NP,I_D) < cr;  % all random numbers < F_CR are 1, 0 otherwise
    FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui
  
    
    %% : Mutate two populations

	pNP = max(round(p_best_rate * I_NP), 2); %% choose at least two best solutions	
	randindex = max(1, ceil(rand(1, I_NP) .* pNP)); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
	pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions

    FM_ui = pbest + sf.*(FM_pm1-FM_popold) + jF.*(FM_pm1 - FM_pm2);
    vi_s0 = FM_popold.*FM_mpo + FM_ui.*FM_mui;
    FM_ui = pbest + sf.*(FM_pm2 - FM_popold) + jF.*(FM_pm3 - FM_pm4);
    vi_s1 = FM_popold.*FM_mpo + FM_ui.*FM_mui;

    %% : Check Constrains    
    vi_s0 = checkConstrain(vi_s0,minPositionsMatrix,maxPositionsMatrix,check,FM_popold);     
    vi_s1 = checkConstrain(vi_s1,minPositionsMatrix,maxPositionsMatrix,check,FM_popold);       
    
    
    %% Evaluate two pop    
    [children_fitness_s0, ~] = feval(fnc,vi_s0,caseStudyData,otherParameters);
    nfes = nfes + I_NP;
    gen = gen + 1;
    for i = 1 : I_NP
        if children_fitness_s0(i) < bsf_fit_var
            bsf_fit_var = children_fitness_s0(i);
            bsf_solution = vi_s0(i, :);
        end
    end
    solMinVector(gen, : )= bsf_solution; 
    fitMaxVector(1,gen) = bsf_fit_var;
    
    
    [children_fitness_s1, ~] = feval(fnc,vi_s1,caseStudyData,otherParameters);
    nfes = nfes + I_NP;
    gen = gen + 1;
    for i = 1 : I_NP
        if children_fitness_s1(i) < bsf_fit_var
            bsf_fit_var = children_fitness_s1(i);
            bsf_solution = vi_s1(i, :);
        end
    end
    solMinVector(gen, : )= bsf_solution;  
    FVr_bestmemit = bsf_solution; 
    fitMaxVector(1,gen) = bsf_fit_var;

    %% : Select the best of the two populations
    aa = children_fitness_s0 < children_fitness_s1 ;    
    if nnz(aa) > I_NP /2
            strategy =0;
            [fitness, I] = min([fitness, children_fitness_s0], [], 2);
            pop(I == 2, :) = vi_s0(I == 2, :);
            
        else
            strategy=1;
            [fitness, I] = min([fitness, children_fitness_s1], [], 2);
            pop(I == 2, :) = vi_s1(I == 2, :);
    end
    
    %% 	Deepening stage: Using the best strategy, 8 modifications are made 
    for j = 1:8           
        [ ~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(I_NP, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);

        cr = normrnd(mu_cr, 0.1);
        cr(mu_cr == -1) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        %% for generating scaling factor
        sf = normrnd(mu_sf, 0.1);
        pos = find(sf <= 0);

        while ~ isempty(pos)
            sf(pos)= normrnd(mu_sf, 0.1);
            pos = find(sf <= 0);
        end

        if(nfes < I_evalmax * 0.5)
            sf = min(sf, ch + 0.1); %%limit the mutation factor value to a maximum
        else
            sf = min(sf, ch + 0.3); %%limit the mutation factor value to a maximum
        end

        jF = ch * sf;

        %%%%%%%%%%%%%%%%%%%
        FM_popold = pop;                  % save the old population
        FVr_ind = randperm(4);               % index pointer array
        FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors

        FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1       

        FM_mui = rand(I_NP,I_D) < cr;  % all random numbers < F_CR are 1, 0 otherwise
        FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui
        pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions


        switch (strategy)
            case 0
                FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions
                FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations
                FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
                FM_ui = pbest + sf.*(FM_pm1-FM_popold) + jF.*(FM_pm1 - FM_pm2);
                vi = FM_popold.*FM_mpo + FM_ui.*FM_mui;     
            case 1
                FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions
                FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations
                FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
                FVr_a3  = randperm(I_NP);
                FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);
                FVr_a4  = FVr_a1(FVr_rt+1);
                FM_pm3 = archive.pop(FVr_a3,:);             % shuffled archived 3
                FM_pm4 = archive.pop(FVr_a4,:);             % shuffled archived 4
                FM_ui = pbest + sf.*(FM_pm2 - FM_popold) + jF.*(FM_pm3 - FM_pm4);
                vi = FM_popold.*FM_mpo + FM_ui.*FM_mui;

        end

        %% Check constrains
        vi=checkConstrain(vi,minPositionsMatrix,maxPositionsMatrix,check,FM_popold);        

        if(nfes + I_NP >= I_evalmax)
            break;
        end
        [children_fitness, ~] = feval(fnc,vi,caseStudyData,otherParameters);
        gen = gen + 1;
        nfes = nfes + I_NP;

        for i = 1 : I_NP
            if children_fitness(i) < bsf_fit_var
                bsf_fit_var = children_fitness(i);
                bsf_solution = vi(i, :);
            end
        end		  

        I = (fitness > children_fitness);
        goodCR = cr(I == 1);  
        badCR = cr(I == 0);  
        goodF = sf(I == 1);
        badF = sf(I == 0);  
        archive = updateArchive(archive, FM_popold(I == 1, :), fitness(I == 1));
        [fitness, I] = min([fitness, children_fitness], [], 2);

        %% : Selection
        pop(I == 2, :) = vi(I == 2, :);

        num_success_params = numel(goodCR);
        num_fail_params = numel(badCR);



        if num_success_params > num_fail_params 
            sum_cr = sum(goodCR);
            sum_sf = sum(goodF);
            sum_cr = sum_cr / num_success_params;
            sum_sf = sum_sf / num_success_params;

            %% for updating the memory of scaling factor 
            memory_sf(memory_pos) = (1.0+ch) * sum_sf - ch * memory_sf(memory_pos);
            memory_cr(memory_pos) = (1.0+ch) * sum_cr - ch * memory_cr(memory_pos);
            %% for updating the memory of crossover rate
            if(memory_cr(memory_pos) <= 0 || memory_cr(memory_pos) > 1)
              memory_cr(memory_pos) = sum_cr;
            end
            if(memory_sf(memory_pos) <= 0 || memory_sf(memory_pos) > 1)
                memory_sf(memory_pos) = sum_sf;
            end           
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size 
                memory_pos = 1; 
            end
        else
            old_sf = memory_sf(memory_pos);
            old_cr = memory_cr(memory_pos);
            sum_cr = sum(goodCR)+sum(badF);
            sum_sf = sum(goodF)+sum(badF);
            sum_cr = sum_cr / I_NP;
            sum_sf = sum_sf / I_NP;

            %% for updating the memory of scaling factor 
            memory_sf(memory_pos) = old_sf + 0.2 * (0.5 * sum_sf - 0.8 * memory_sf(memory_pos)  );
            memory_cr(memory_pos) = old_cr + 0.2 * (0.5 * sum_cr - 0.8 * memory_cr(memory_pos) );
            if(memory_cr(memory_pos) <= 0 || memory_cr(memory_pos) > 1)
              memory_cr(memory_pos) = old_cr;
            end
            if(memory_sf(memory_pos) <= 0 || memory_sf(memory_pos) > 1)
                memory_sf(memory_pos) = old_sf;
            end            
        end
        if(num_success_params == 0 && fitMaxVector(1,gen-1) == bsf_fit_var && nfes > I_evalmax * 0.4)
            archive.pop(randi(I_NP)) = solMinVector(num_reset);
            num_reset = num_reset +1;            
        end
        FVr_bestmemit = bsf_solution; 
        solMinVector(gen, : )= bsf_solution; 
        fitMaxVector(1,gen) = bsf_fit_var;
        fprintf('Fitness value: %f\n',bsf_fit_var);%%fitMaxVector(1,gen) )
        fprintf('Eval: %d Strategy: %d, Check %d\n',nfes,strategy,check);        
    end       
end
   
Fit_and_p=[fitMaxVector(1,gen-1) 0];

function p=checkConstrain(p,lowMatrix,upMatrix,BRM,FM_base)
switch BRM
    case 1 %Our method
        [idx] = find(p<lowMatrix);
        p(idx)=lowMatrix(idx);
        [idx] = find(p>upMatrix);
        p(idx)=upMatrix(idx);
    case 2
        [idx] = find(p<lowMatrix);
        randoom = rand(size(p,1),size(p,2));
        resultado = (FM_base -lowMatrix);
        p(idx)= lowMatrix(idx) + resultado(idx) .* randoom (idx);

        [idx] = find(p>upMatrix);
        randoom = rand(size(p,1),size(p,2));
        resultado = (upMatrix - FM_base);
        p(idx)= FM_base(idx) + resultado(idx) .* randoom (idx);
    case 3 %Random reinitialization
        [idx] = [find(p<lowMatrix);find(p>upMatrix)];
        replace=unifrnd(lowMatrix(idx),upMatrix(idx),length(idx),1);
        p(idx)=replace;
    case 4 %Bounce Back
      [idx] = find(p<lowMatrix);
      p(idx)=unifrnd(lowMatrix(idx),FM_base(idx),length(idx),1);
        [idx] = find(p>upMatrix);
      p(idx)=unifrnd(FM_base(idx), upMatrix(idx),length(idx),1);
end
