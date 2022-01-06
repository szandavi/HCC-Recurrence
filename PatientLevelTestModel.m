clc;
clear;
%%
load('Final_SVM_10Fold_SelectedFeatures.mat')
[a,b] = max(vertcat(Result.acc));

r_features = Result(b).r_features;
Model = Result(b).model;
%%
file_name = 'final_input.csv';
opts = detectImportOptions(file_name);
X_ML = readtable(file_name,opts);

%%
id_NoNan = ~isnan(X_ML.Recurrence);

Data = X_ML(id_NoNan,:);

DaysToRec = Data.DaysToRecurrence;

%% censor cut-off

Censor = 365;

for j = 1:1
    censor = Censor(j);
    disp(censor)
    %%
    idx = DaysToRec <= censor;
    
    id_censor = idx;
    
    data = Data(~id_censor,:);
   
    %% Features
    id_features = @(x)(strcmp(x,data.Properties.VariableNames));
    
    list_selected_features = {'INR','Satellite','Sex','No_Lesions',...
        'Cirrhosis','Bilirubin',...
        'LiverDisease','Size','PlateletCount',...
        'MetabolicRiskFactors','eGFR',...
        'Albumin','CVS'...
        'BMI','ALT'};
    
    Features_RF_KM = list_selected_features;
    
    T_X_ML_Sel = [];
    
    for i = 1:length(Features_RF_KM)+1
        if i == 1
            indx = id_features('Recurrence');
        else
            indx = id_features(Features_RF_KM{i-1});
        end
        
        T_X_ML_Sel = [T_X_ML_Sel data(:,indx)];
    end
    
    %%
    XX_ML = T_X_ML_Sel;
    %%
    % MetabolicRiskFactors > 2 ; No_Lensions >=2, eGFR>=90, ALT > 50,
    % BMI > 25, AFP > 100 (<=8),  sizeOfLargestLesion_cm_ >=5, albumin>=35,
    % Age > 65, bili>20,
    % PlateletCount>=150, INR> 1.1,
    
    List_thr = {'MetabolicRiskFactors','No_Lensions','eGFR','ALT',...
        'BMI','Size','Albumin','Age','Bilirubin','PlateletCount','INR'};
    
    indxFinder = @(x)(strcmp(x,XX_ML.Properties.VariableNames));
    
    idx = cellfun(indxFinder,List_thr,'UniformOutput',false);
    idx_tr = vertcat(idx{:});
    
    %=========== if AFP
    id_AFP = indxFinder('AFP');
    
    id_con_100 = XX_ML{:,id_AFP} > 100;
    id_con_8_100 = XX_ML{:,id_AFP} > 8 & XX_ML{:,id_AFP} <= 100;
    id_con_8 = XX_ML{:,id_AFP} <= 8;
    
    XX_ML{id_con_100,id_AFP} = 2;
    XX_ML{id_con_8_100,id_AFP} = 1;
    XX_ML{id_con_8,id_AFP} = 0;
    %==================================================
    
    condition = [2 1.9 89 50 25 4.9 34 65 20 149 1.1];
    for tr_list = 1:length(List_thr)
        if ~isempty(XX_ML.Properties.VariableNames(idx_tr(tr_list,:)))
            
            id_con_1 = XX_ML{:,idx_tr(tr_list,:)} > condition(tr_list);
            id_con_0 = XX_ML{:,idx_tr(tr_list,:)} <= condition(tr_list);
            
            XX_ML{id_con_1,idx_tr(tr_list,:)} = 1;
            XX_ML{id_con_0,idx_tr(tr_list,:)} = 0;
            
        end
    end
    
    %% impute missing value
    % shuffle rows
    r_initial = randperm(size(XX_ML,1));
    XX_ML = XX_ML(r_initial,:);
    XX_All = XX_ML{:,2:end};
    
    imputedData = knnimpute(XX_All');
    
    XX_ML{:,2:end} = round(imputedData');
    
    %%
    X_raw = XX_ML{:,2:end}';
    
    % The raw data is categorical data, so the normalization is not
    % required
    
    x_norm = X_raw;
    
    Y = XX_ML{:,1};
    
    %%
    ERR = [];
    ACC = [];
    SEN = [];
    SPE = [];
    F1 = [];
    Y_Patient = [];
    %%
    for i = 1:500
        
        r_sample = randperm(size(x_norm,2));
        
        SampleSize = size(x_norm,2);
        
        Y_model = Y(r_sample);
        
        X_train = x_norm(:,r_sample);
        X_Raw = X_raw(:,r_sample);
        
%         r_features = randperm(size(x_norm,1));
        
        X_train = X_train(r_features,:);
        X_Raw = X_Raw(r_features,:);
        %
        %% ML model
        x_train_test = X_train;
        Y_train_test = cellstr(num2str(Y_model));
        
        Y_train_test(strcmp(Y_train_test,'1'))= {'Yes'};
        Y_train_test(strcmp(Y_train_test,'0'))= {'No'};
        
        cv = cvpartition(size(x_train_test,2),'HoldOut',0.1);
        idx = cv.test;
        
        % we picked all data to do the 10 fold cross validation
        XTrain = x_train_test;%(:,~idx);
        XTest  = x_train_test(:,idx);
        
        YTrain = Y_train_test;%(~idx);
        YTest  = Y_train_test(idx);
        
        [count,label] = hist(categorical(Y_train_test),unique(Y_train_test));
        
        %%
        % n = 1: RF, 2: KNN, 3: SVM
        
        % RF
        n = 3;
        
        disp(i);
        
        pred_test = [];
        
        pred_train = predict(Model,XTrain');
        
        Y_1_0 = strcmp(YTrain,pred_train);
        
        cp_train = classperf(YTrain, pred_train);
        
        err = cp_train.ErrorRate;
        acc_train = cp_train.CorrectRate;
        acc =  cp_train.CorrectRate;
        sen =  cp_train.Sensitivity;
        spe =  cp_train.Specificity;
        ppv = cp_train.PositivePredictiveValue;
        [cm,order] = confusionmat(YTrain,pred_train);

        f1 = 2*(ppv*sen)/(ppv+sen);
        
        %%
        ERR = [ERR;err];
        ACC = [ACC;acc];
        SEN = [SEN;sen];
        SPE = [SPE;spe];
        F1 = [F1;f1];
        Y_Patient = [Y_Patient Y_1_0(sort(r_sample))];
        
        
    end
    
end
%%
% save('Best_SVM_10Fold_SelectedFeatures','Result')
