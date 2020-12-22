%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chaotic Evolutionary Particle Swarm Optimization
% Author: Phillipe Vilaça Gomes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Fit_and_p, sol, fitMaxVector, Best_otherInfo] = ...
    EPSOu(epsouParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,No_eval_Scenarios)

%-----This is just for notational convenience and to keep the code uncluttered.--------

I_NP         = epsouParameters.I_NP;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
I_itermax    = epsouParameters.stop;
replic = epsouParameters.r;

wors = round(epsouParameters.selecWorse*I_NP/100);

%-----Initialize population and some arrays-------------------------------

minPositionsMatrix=repmat(low_habitat_limit,I_NP,1);
maxPositionsMatrix=repmat(up_habitat_limit,I_NP,1);
populacao = unifrnd(minPositionsMatrix,maxPositionsMatrix,I_NP,I_D);

%-----Evaluation of the first population and parameters--------------------

[solFitness_M,solPenalties_M, Struct_Eval] = ...
    fitnessFun_DER_WCCI(populacao,caseStudyData,otherParameters,No_eval_Scenarios);

MedDesv = mean(solFitness_M')+std(solFitness_M');
MedDesv=MedDesv';

PenDesv = mean(solPenalties_M')+std(solPenalties_M');
PenDesv = PenDesv';

[a1,b1] = sort (MedDesv);
I_best_index = b1(1);
pbest = populacao;
gbest = populacao(b1(1),:);

Vpbest = MedDesv;
Vgbest = MedDesv(b1(1));
Vgbestpen = PenDesv(b1(1));

iteration = 1;
GVEST(iteration,1) = Vgbest;

%%

fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %

%----------------------EPSO Iterative Process------------------------------

wi1 = rand(I_NP,replic);
wi2 = rand(I_NP,replic);
wi3 = rand(I_NP,replic);
wi4 = rand();

Vel11 = rand(I_D,I_NP);

while iteration<I_itermax
    iteration = iteration + 1;
    %% Replication Block
    for rep=1:replic
        POP{rep}=populacao;
    end
    
    %% Mutation Block
    wi1 = (0.5+rand-(1./(1+exp(wi1))));
    wi2 = (0.5+rand-(1./(1+exp(wi2))));
    wi3 = (0.5+rand-(1./(1+exp(wi3))));
    wi4 = (0.5+rand-(1./(1+exp(wi4))));
    Kbest1 = gbest + round(2*wi4-1);
    
    winew = wmax-((wmax-wmin)/I_itermax)*iteration;
    
    %% Recombination Block
    for rep=1:replic
        % EPSO (mutated weights chaotic evolution)
        for i=1:I_NP
            Vel11(:,i)=Vel11(:,i).*wi1(i,1)+wi2(i,1).*(pbest(i,:)-POP{1,rep}(i,:))'+wi3(i,1).*(Kbest1-POP{1,rep}(i,:))';
            POP{1,rep}(i,:) = POP{1,rep}(i,:)+Vel11(:,i)';
            % Verificação do limite superior
            vetAn = POP{1,rep}(i,:);
            vetAn(vetAn>up_habitat_limit)=up_habitat_limit(vetAn>up_habitat_limit);
            POP{1,rep}(i,:)=vetAn;
            POP{1,rep}(i,8:14)=round(POP{1,rep}(i,8:14));
            % Verificação do limite inferior
            vetAn = POP{1,rep}(i,:);
            vetAn(vetAn<low_habitat_limit)=low_habitat_limit(vetAn<low_habitat_limit);
            POP{1,rep}(i,:)=vetAn;
            POP{1,rep}(i,8:14)=round(POP{1,rep}(i,8:14));
        end
    end
    
    %% Evaluation block
    for rep=1:replic
        [solFitness_MPOP{rep},solPenalties_MPOP{rep}, Struct_Eval_MPOP{rep}] = ...
            fitnessFun_DER_WCCI(POP{1,rep},caseStudyData,otherParameters,No_eval_Scenarios);
    end
        
    
    %% Selection Block
    
    for rep=1:replic
        MedDesvPOP{rep} = mean(solFitness_MPOP{1,rep}')+std(solFitness_MPOP{1,rep}');
        MedDesvPOP{rep}=MedDesvPOP{rep}';
        PenDesv = mean(solPenalties_MPOP{1,rep}')+std(solPenalties_MPOP{1,rep}');
        pbest(MedDesvPOP{rep}<Vpbest,:)= POP{1,rep}(MedDesvPOP{rep}<Vpbest,:);
        Vpbest(MedDesvPOP{rep}<Vpbest)=MedDesvPOP{rep}(MedDesvPOP{rep}<Vpbest);
        [a1,b1] = sort (MedDesvPOP{rep});
        VbestP=a1(1);
        if VbestP<Vgbest
            Vgbest=VbestP;
            gbest=POP{1,rep}(b1(1),:);
            Vgbestpen=PenDesv(b1(1));
            % Other results (worse performance)
            [S_val, worstS]=max(solFitness_MPOP{rep},[],2);
            solFitness_M = solFitness_MPOP{rep};
            solPenalties_M = solPenalties_MPOP{rep};
            Struct_Eval = Struct_Eval_MPOP{rep};
            [~,I_best_index] = min(S_val);
            Best_otherInfo.idBestParticle = I_best_index;
            Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
            Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
            Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
            Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
            Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
            Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
            Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
            Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
            Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
        end
    end
    
    
    
    MatrizMedDesvPop = [MedDesvPOP{1} MedDesvPOP{2} MedDesvPOP{3}];
    
    [valoresM,indiceM] = sort(MatrizMedDesvPop');
    
    Melhores = [valoresM(1,:);indiceM(1,:)]';
    Piores = [valoresM(replic,:);indiceM(replic,:)]';
    Setwors = randi([1,I_NP],1,wors);
    
    Melhores(Setwors,:)=Piores(Setwors,:);
    newpop = populacao;
    newfitpop = MedDesv;
    
    for indix=1:I_NP
        newpop(indix,:)=POP{1,Melhores(indix,2)}(indix,:);
        newfitpop(indix)=MedDesvPOP{1,Melhores(indix,2)}(indix);
    end
    
    %% Important Results
    
    
    GVEST(iteration,1) = Vgbest;
    PENBES(iteration,1) = Vgbestpen;
    
    %% Store variables
    sol = gbest;
    
    %% New population
    populacao = newpop;
    MedDesv = newfitpop;
    
    
end

%% Final Results
fitMaxVector=[GVEST PENBES]';
Fit_and_p = fitMaxVector(:,iteration)';
sol = gbest;


end




