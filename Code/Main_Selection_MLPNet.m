clc;
clear;
%%
data_train = load('Data_train_after_binary.mat');
data_test = load('Data_test_after_binary.mat');

data_train = data_train.data_train;
data_test = data_test.data_test;

%%
Data_train = data_train;
Data_test = data_test;

%% Test Data

for j = 1:1
    %%

    DaysToRec = Data_train.DaysToRecurrence;
    DaysToRec_test = Data_test.DaysToRecurrence;

    %% censor cut-off

    Censor = 365;


    censor = Censor(1);
    disp(censor)
    %%
    idx = DaysToRec <= censor;
    idx_test = DaysToRec_test <= censor;

    id_censor = idx;

    data = Data_train(~id_censor,:);
    data_test = Data_test;
    %% Features
    id_features = @(x)(strcmp(x,data.Properties.VariableNames));
%     
    list_selected_features = {'INR','LiverDisease','No_Lesions',...
        'Ethnicity','Cirrhosis','DM','HPVG','ALT',...
        'eGFR','Albumin','AFP','BMI','Satellite',...
        'LVI','Sex','Bilirubin','Age','IHD','Size','PriorTACE'}; 

    Features_RF_KM = list_selected_features;
    
    T_X_ML_Sel = [];
    T_X_ML_Sel_test = [];
    
    for i = 1:length(Features_RF_KM)+1
        if i == 1
            indx = id_features('Recurrence');
        else
            indx = id_features(Features_RF_KM{i-1});
        end
        
        T_X_ML_Sel = [T_X_ML_Sel data(:,indx)];
        T_X_ML_Sel_test = [T_X_ML_Sel_test data_test(:,indx)];
    end
    
    %%
    XX_ML = T_X_ML_Sel;
    XX_ML_test = T_X_ML_Sel_test;
    %% tsne
%     y = tsne(XX_ML{:,2:end},'Algorithm','Exact','NumPCAComponents',10,statset('MaxIter',2000));
%     gscatter(y(:,1),y(:,2),XX_ML{:,1});
    %%
    X_raw = XX_ML{:,2:end}';
    X_test_china = XX_ML_test{:,2:end}';
    
    % The raw data is categorical data, so the normalization is not
    % required

    x_norm = X_raw;
    
    Y = XX_ML{:,1};
    Y_china = XX_ML_test{:,1};

    X_test = X_test_china;
    
    %%
    j = 1;
    for i = 1:100
        
        r_sample = randperm(size(x_norm,2));
        
        SampleSize = size(x_norm,2);
        
        Y_model = Y(r_sample);
        
        X_train = x_norm(:,r_sample);
        X_Raw = X_raw(:,r_sample);
        
        r_features = randperm(size(x_norm,1));
%         
        X_train = X_train(r_features,:);
        X_Raw = X_Raw(r_features,:);
        X_test_china = X_test(r_features,:);
        %
        %% ML model
        x_train_test = X_train;
        Y_train_test = cellstr(num2str(Y_model));

        Y_test_china = cellstr(num2str(Y_china));
        
        Y_train_test(strcmp(Y_train_test,'1'))= {'Yes'};
        Y_train_test(strcmp(Y_train_test,'0'))= {'No'};

        Y_test_china(strcmp(Y_test_china,'1'))= {'Yes'};
        Y_test_china(strcmp(Y_test_china,'0'))= {'No'};
        
        cv = cvpartition(size(x_train_test,2),'HoldOut',0.2);
        idx = cv.test;
        
        % we picked all data to do the 10 fold cross validation
        XTrain = x_train_test(:,~idx);
        XVal  = x_train_test(:,idx);
    
        YTrain = Y_train_test(~idx);
        YVal  = Y_train_test(idx);
        
        [count,label] = hist(categorical(Y_train_test),unique(Y_train_test));
        %%

        ddata.training.input = XTrain';
        ddata.training.output = onehotencode(categorical(YTrain),2);

        % Test Data
        ddata.test.input = X_test_china';
        ddata.test.output = onehotencode(categorical(Y_test_china),2);
        
        % Validation Data
        ddata.validation.input = XVal';
        ddata.validation.output = onehotencode(categorical(YVal),2);

        % define input and output features
        ddata.input_count = size(XTrain,1) ;
        ddata.output_count = 2;
        ddata.training_count = size(XTrain,2);   % Number of samples in training set
        ddata.test.count = size(X_test_china,2);

        net = MLP_Model(ddata);
%% Test
        pre_test = net.test(X_test_china');

        for ii=1:size(pre_test)
            if pre_test(ii,1) > pre_test(ii,2)
                PreTest{ii} = 'No';
                Y_pred_roc(ii) = 0;
            else
                PreTest{ii} = 'Yes';
                Y_pred_roc(ii) = 1;
            end
        end

        cp_test = classperf(Y_test_china, PreTest,'Positive',{'Yes'},'Negative',{'No'});
        
        err_test = cp_test.ErrorRate;
        acc_test =  cp_test.CorrectRate;
        sen_test =  cp_test.Sensitivity;
        spe_test =  cp_test.Specificity;
        ppv_test = cp_test.PositivePredictiveValue;
        [cm_test,order_test] = confusionmat(Y_test_china,PreTest);

        f1_test = 2*(ppv_test*sen_test)/(ppv_test+sen_test);

%% Train
        pre_train = net.test(XTrain');

        for ii=1:size(pre_train)
            if pre_train(ii,1) > pre_train(ii,2)
                PreTrain{ii} = 'No';
                Y_pred_roc_train(ii) = 0;
            else
                PreTrain{ii} = 'Yes';
                Y_pred_roc_train(ii) = 1;
            end
        end

        cp_train = classperf(YTrain, PreTrain,'Positive',{'Yes'},'Negative',{'No'});
        
        err_train = cp_train.ErrorRate;
        acc_train =  cp_train.CorrectRate;
        sen_train =  cp_train.Sensitivity;
        spe_train =  cp_train.Specificity;
        ppv_train = cp_train.PositivePredictiveValue;
        [cm_train,order_train] = confusionmat(YTrain,PreTrain);

        f1_train = 2*(ppv_train*sen_train)/(ppv_train+sen_train);

        %%

        Result(i,j).err_train = err_train;
        Result(i,j).acc_train = acc_train;
        Result(i,j).sen_train = sen_train;
        Result(i,j).spe_train = spe_train;
        Result(i,j).f1_train = f1_train;
        Result(i,j).cm_train = cm_train;
        Result(i,j).order_train = order_train;

        Result(i,j).err_test = err_test;
        Result(i,j).acc_test = acc_test;
        Result(i,j).sen_test = sen_test;
        Result(i,j).spe_test = spe_test;
        Result(i,j).f1_test = f1_test;
        Result(i,j).cm_test = cm_test;
        Result(i,j).order_test = order_test;
        
        Result(i,j).r_sample = r_sample;
        Result(i,j).r_features = r_features;


        Result(i,j).net = net;
        

        Result(i,j).ypred_train = pre_train;
        Result(i,j).ypred_test = pre_test;
        
        clear net ddata
    
    end
end

%%
% save('SelectedFeatures_MLP_Model','Result')
