close all; clear all; clc;

%% parameters
parameter.networkSize = 20;   % node number
parameter.averageDegree = 4;  % average degree of the net
parameter.changeRate = 1;    % change Rate, creating and deleting links number
parameter.sampleRate = 3;    % obeservation number in each period
parameter.times = 20;        % network evolving times

parameter.mode = 1; % Data generation   1:Ultimatum game   2£ºLorentz
parameter.networkMode = 1; % Initial Network   1:Homogeneous net  2:BA scale-free
parameter.noise = 0;    % noise?     0: No  || other value:Standard deviation (i.g. 0.3)

parameter.sampleModel = 0; % 0 even sample      1 uneven sample
parameter.weightModel = 0; % 0 unweighted net      1 weighted net

%% data generation
[observation, strategy, Adjset, straSeries, sampleMat] = dataGeneration(parameter);

%% identification
[thetaOne4one] = identificationOne(observation, strategy, [0.0001,0.5716]);        % One for One 
[thetaAll4one] = identificationAll(observation, strategy, [0.0001,0.3]);   % All for One
[thetaATNISD] = identificationATNISD(observation, strategy,[0.00001, 0.3, 0.00001],100);% The proposed method

%% evaluation
networkSize = parameter.networkSize;
times = parameter.times;target = reshape(Adjset, networkSize^2 * times,1);
theta = thetaOne4one;
score = reshape(theta, networkSize^2 * times, 1);
target = reshape(Adjset, networkSize^2 * times,1);
meanSquare = mean(abs(score-target));
fprintf('OO Error is: %s\n', meanSquare)
[AUROC_One4one,AUPR_One4one,prec_One4one, tpr_One4one, fpr_One4one, thresh] = prec_rec(score, target, 'holdFigure',1);
prec_rec(score, target, 'holdFigure',1);


theta = thetaAll4one;
score = reshape(theta, networkSize^2 * times, 1);
meanSquare = mean(abs(score-target));
fprintf('AO Error is: %s\n', meanSquare)
[AUROC_All4one,AUPR_All4one,prec_All4one, tpr_All4one, fpr_All4one, thresh] = prec_rec(score, target, 'holdFigure',1);
prec_rec(score, target, 'holdFigure',2);


theta = thetaATNISD;
score = reshape(theta, networkSize^2 * times, 1);
meanSquare = mean(abs(score-target));
fprintf('ATNISD Error is: %s\n', meanSquare)
[AUROC_ATNISD,AUPR_ATNISD,prec_ATNISD, tpr_ATNISD, fpr_ATNISD, thresh] = prec_rec(score, target, 'holdFigure',1);
prec_rec(score, target, 'holdFigure',2);
