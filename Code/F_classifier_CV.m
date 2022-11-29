function [err,acc_train,acc,sen,spe,ppv,AUC,order,...
    cm,model,pred_train] = F_classifier_CV(xtrain,ytrain,...
    n,optimize,Positive_string)

XTrain = xtrain;
YTrain = ytrain;

auto_optimize = optimize;

%'randomsearch' 'gridsearch' 'bayesopt'
Optimizer = 'randomsearch';

Max_iteration = 60;
ShowPlots = false;
UseParallel = true;

if auto_optimize

    % If no pool, do not create new one.
    poolobj = gcp('nocreate');

    if isempty(poolobj)
        parpool('local');
        %         parpool(2);
    end
end


switch n


    case 1 % Random Forest
        if auto_optimize
            t = templateTree('Surrogate','on','NumVariablesToSample','all',...
                'PredictorSelection','curvature','Reproducible',true);
       
            Model = fitcensemble(XTrain,YTrain,'OptimizeHyperparameters',...
                'all','Learners',t, ...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel,'AcquisitionFunctionName','expected-improvement-plus'));

        else

            t = templateTree('MaxNumSplits',2,'MinLeafSize',3,...
                'SplitCriterion','deviance');
            numTrees = 29;

            Method = 'AdaBoostM1';

            Model = fitcensemble(XTrain,YTrain,'Learners',t, ...
                'NumLearningCycles',numTrees,'LearnRate',0.31616, ...
                'Method',Method);

        end


    case 2 % KNN

        if auto_optimize
            Model = fitcknn(XTrain,YTrain,'OptimizeHyperparameters','all',...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
        else
            NumNeighbors = 8;
            Distance = 'euclidean';
            BreakTies = 'nearest';
            Model = fitcknn(XTrain,YTrain,'NumNeighbors',NumNeighbors,...
                'NSMethod','exhaustive','Distance',Distance,...
                'BreakTies',BreakTies,'Standardize',true,'DistanceWeight','inverse');

        end


    case 3 % SVM

        if auto_optimize

            t = templateSVM('SaveSupportVectors',true,'BoxConstraint',1,'KernelScale','auto');
            
            Model = fitcecoc(XTrain,YTrain,'OptimizeHyperparameters',{'Coding',...
                'Standardize','KernelScale','PolynomialOrder','KernelFunction'},'Learners',t, ...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,...
                'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
            
        else
            t = templateSVM('Standardize',false,'SaveSupportVectors',true,...
                'BoxConstraint',1,'KernelFunction','gaussian','KernelScale',2.1147);

            Model = fitcecoc(XTrain,YTrain,'Learners',t,'Coding','onevsall');

        end

    case 4 % alternative

        if auto_optimize
            t = templateLinear();

            Model = fitcecoc(XTrain,YTrain,'OptimizeHyperparameters','all','Learners',t, ...
                'HyperparameterOptimizationOptions',...
                struct('Optimizer',Optimizer,...
                'MaxObjectiveEvaluations',Max_iteration,...
                'ShowPlots',ShowPlots,'UseParallel',UseParallel));
        else
            t = templateSVM('Standardize',false,'SaveSupportVectors',true,...
                'BoxConstraint',1,'KernelFunction','gaussian','KernelScale',2.1147);

            Model = fitcecoc(XTrain,YTrain,'Learners',t,'Coding','onevsall');

        end

end


CVModel = crossval(Model,'kfold',10);
[pred_train,scores] = kfoldPredict(CVModel);
% pred_train = predict(Model,XTrain);

[Class,~] = unique(YTrain);
Positive = Class(strcmp(Class,Positive_string));
Negative = Class(~strcmp(Class,Positive_string));
%
if isempty(Positive_string)
    cp_train = classperf(YTrain, pred_train);
else
    cp_train = classperf(YTrain, pred_train,'Positive',Positive,'Negative',Negative);
end

err = cp_train.ErrorRate;
acc_train = cp_train.CorrectRate;
acc =  cp_train.CorrectRate;
sen =  cp_train.Sensitivity;
spe =  cp_train.Specificity;
ppv = cp_train.PositivePredictiveValue;
[cm,order] = confusionmat(YTrain,pred_train);

%%

if isempty(Positive_string)
    AUC = [];
else
    innxx = strcmp(CVModel.ClassNames,Positive_string);
%     mdlSVM = fitPosterior(Model);
%     [~,score_svm] = resubPredict(mdlSVM);

    [X_auc,Y_auc,~,auc,OPTROCPT] = perfcurve(YTrain,scores(:,innxx),Positive_string);
    AUC.X_auc = X_auc;
    AUC.Y_auc = Y_auc;
    AUC.auc = auc;
    AUC.OPTROCPT = OPTROCPT;
end

model = {Model,CVModel};

% function end
end
