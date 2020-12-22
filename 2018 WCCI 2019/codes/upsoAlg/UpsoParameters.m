%% our version
deParameters.numOfParticles = 60;
deParameters.max_iterations = 50;
deParameters.Neval = 3;
% deParameters.topology = 1;
deParameters.topology = 2;
deParameters.version = 8;
N = deParameters.max_iterations;
% rmat = [2 , 4, 8, 10 ,16 , 8];
% pos =  [2 2 1 1 2 2].*(N/10);
% deParameters.matrix_r = [rmat(1).*ones(pos(1),1)' ,rmat(2).*ones(pos(2),1)' ,rmat(3).*ones(pos(3),1)' ,rmat(4).*ones(pos(4),1)' ,rmat(5).*ones(pos(5),1)' ,rmat(6).*ones(pos(6),1)'  ];
rmat = [4];
pos =  [10].*(N/10);
deParameters.matrix_r = [rmat(1).*ones(pos(1),1)' ];
% deParameters.mode = 'lol_mode';
deParameters.mode = 'test_mode';
deParameters.numOfDim = 3408;


