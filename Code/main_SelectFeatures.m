clc;
clear;
%%
file_name = 'Raw_Data.xlsx';
%%
opts = detectImportOptions(file_name);
X_ML_Alfred = readtable(file_name,opts,'Sheet',1);
%%
opts = detectImportOptions(file_name,'Sheet',2);
X_ML_Austin = readtable(file_name,opts);
%%
opts = detectImportOptions(file_name,'Sheet',3);
X_ML_China = readtable(file_name,opts);
%%

cv = cvpartition(size(X_ML_China,1),'HoldOut',0.40);

% The accuracy depends on what portion of Nan NAFL will be selected!
X_ML_China_test  = X_ML_China(cv.test,:);
X_ML_China_train = X_ML_China(~cv.test,:);

%%
Data = [X_ML_Alfred;X_ML_Austin;X_ML_China_train];
Data_test = X_ML_China_test;

%% Ethnicity
indxFinder_ethnicity = @(x,property_list)(strcmp(x,property_list));

id_eth_train = indxFinder_ethnicity('Ethnicity',Data.Properties.VariableNames);
id_eth_test = indxFinder_ethnicity('Ethnicity',Data_test.Properties.VariableNames);

id_eth_3 = Data{:,id_eth_train} == 3;
id_eth_4 = Data{:,id_eth_train} == 4;

Data{id_eth_3,id_eth_train} = 0;
Data{id_eth_4,id_eth_train} = 2;
Data{isnan(Data{:,id_eth_train}),id_eth_train} = 2;

id_eth_3 = Data_test{:,id_eth_test} == 3;
id_eth_4 = Data_test{:,id_eth_test} == 4;

Data_test{id_eth_3,id_eth_test} = 0;
Data_test{id_eth_4,id_eth_test} = 2;
Data_test{isnan(Data_test{:,id_eth_test}),id_eth_test} = 2;

%% Sex 
% In Raw_Data, Female is set to 1 because of 
% consistency with the paper; it is changed to 0.

id_sex_train = indxFinder_ethnicity('Sex',Data.Properties.VariableNames);
id_sex_test = indxFinder_ethnicity('Sex',Data_test.Properties.VariableNames);

id_sex_Female = Data{:,id_sex_train} == 1;
Data{id_sex_Female,id_sex_train} = 0;
Data{~id_sex_Female,id_sex_train} = 1;

id_sex_Female = Data_test{:,id_sex_test} == 1;
Data_test{id_sex_Female,id_sex_test} = 0;
Data_test{~id_sex_Female,id_sex_test} = 1;

%% Test Data

for j = 1:1
    %%

    DaysToRec = Data.DaysToRecurrence;
    DaysToRec_test = Data_test.DaysToRecurrence;

    %% censor cut-off

    Censor = 365;
    
    censor = Censor(1);
    disp(censor)
    %%
    idx = DaysToRec <= censor;
    idx_test = DaysToRec_test <= censor;

    id_censor = idx;

    data = Data(~id_censor,:);
    data_test = Data_test(~idx_test,:);

    id_nan = isnan(data.DaysToRecurrence);
    id_nan_test = isnan(data_test.DaysToRecurrence);

    data = data(~id_nan,:);
    data_test = data_test(~id_nan_test,:);

    %% Features
    id_features = @(x)(strcmp(x,data.Properties.VariableNames));
    id_features_test = @(x)(strcmp(x,data_test.Properties.VariableNames));
%     
    list_All_features = {'LiverDisease','Age','PriorTACE','Bilirubin',...
        'Albumin','INR','PlateletCount','AFP','No_Lesions','Size','Satellite',...
        'Cirrhosis','Sex','Ethnicity','LVI','HPVG','ALT','BMI','DM','Hypertension',...
        'eGFR','IHD','CVS'}; 
    
%     list_All_features = data.Properties.VariableNames(2:end);

    Features_RF_KM = list_All_features;
    
    T_X_ML_Sel = [];
    T_X_ML_Sel_test = [];
    
    for i = 1:length(Features_RF_KM)+1
        if i == 1
            indx = id_features('Recurrence');
            indx_test = id_features_test('Recurrence');
        else
            indx = id_features(Features_RF_KM{i-1});
            indx_test = id_features_test(Features_RF_KM{i-1});
        end
        
        T_X_ML_Sel = [T_X_ML_Sel data(:,indx)];
        T_X_ML_Sel_test = [T_X_ML_Sel_test data_test(:,indx_test)];
    end
    
    %%
    XX_ML = T_X_ML_Sel;
    XX_ML_test = T_X_ML_Sel_test;

    %% impute missing value with KNN
    r_initial = randperm(size(XX_ML,1));
    XX_ML = XX_ML(r_initial,:);
    XX_All = XX_ML{:,2:end};

    imputedData = knnimpute(XX_All');

    XX_ML{:,2:end} = imputedData';

    r_initial_test = randperm(size(XX_ML_test,1));
    XX_ML_test = XX_ML_test(r_initial_test,:);
    XX_All_test = XX_ML_test{:,2:end};

    imputedData_test = knnimpute(XX_All_test');

    XX_ML_test{:,2:end} = imputedData_test';

    data_test_save_after_imput  =  data_test;
    data_train_save_after_imput = data;
    
    data_test_save_after_imput = data_test_save_after_imput(r_initial_test,:);
    data_train_save_after_imput = data_train_save_after_imput(r_initial,:);

    data_test_save_after_imput{:,6:end} = XX_ML_test{:,2:end};
    data_train_save_after_imput{:,6:end} = XX_ML{:,2:end};
    
    %% Binarization
    % No_Lensions >1, eGFR>=90, ALT > 50,
    % BMI >= 25, AFP > 100 (<=8),  Size >=5, albumin<35,
    % Age >= 65, bili>20,
    % PlateletCount<150, INR> 1.1,
    
    indxFinder = @(x)(strcmp(x,XX_ML.Properties.VariableNames));

    %=========== if AFP
    id_AFP = indxFinder('AFP');
    XX_ML = AFP_Binary(id_AFP,XX_ML,[],false);
    XX_ML_test = AFP_Binary(id_AFP,XX_ML_test,[],false);

    % ============== if Albumin
    % Train
    id_Alb = indxFinder('Albumin');
    id_con_1 = XX_ML{:,id_Alb} < 35;
    XX_ML{id_con_1,id_Alb} = 1;
    XX_ML{~id_con_1,id_Alb} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_Alb} < 35;
    XX_ML_test{id_con_1,id_Alb} = 1;
    XX_ML_test{~id_con_1,id_Alb} = 0;

    % ============== if PlateletCount
    id_pc = indxFinder('PlateletCount');
    id_con_1 = XX_ML{:,id_pc} < 150;
    XX_ML{id_con_1,id_pc} = 1;
    XX_ML{~id_con_1,id_pc} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_pc} < 150;
    XX_ML_test{id_con_1,id_pc} = 1;
    XX_ML_test{~id_con_1,id_pc} = 0;

    % ============== if eGFR
    id_gfr = indxFinder('eGFR');
    id_con_1 = XX_ML{:,id_gfr} < 90;
    XX_ML{id_con_1,id_gfr} = 1;
    XX_ML{~id_con_1,id_gfr} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_gfr} < 90;
    XX_ML_test{id_con_1,id_gfr} = 1;
    XX_ML_test{~id_con_1,id_gfr} = 0;

    % ============== if No_Lesions
    id_nLen = indxFinder('No_Lesions');
    id_con_1 = XX_ML{:,id_nLen} > 1;
    XX_ML{id_con_1,id_nLen} = 1;
    XX_ML{~id_con_1,id_nLen} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_nLen} > 1 ;
    XX_ML_test{id_con_1,id_nLen} = 1;
    XX_ML_test{~id_con_1,id_nLen} = 0;

    % ============== if ALT
    id_alt = indxFinder('ALT');
    id_con_1 = XX_ML{:,id_alt} > 50;
    XX_ML{id_con_1,id_alt} = 1;
    XX_ML{~id_con_1,id_alt} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_alt} > 50 ;
    XX_ML_test{id_con_1,id_alt} = 1;
    XX_ML_test{~id_con_1,id_alt} = 0;

    % ============== if BMI
    id_bmi = indxFinder('BMI');
    id_con_1 = XX_ML{:,id_bmi} >= 25;
    XX_ML{id_con_1,id_bmi} = 1;
    XX_ML{~id_con_1,id_bmi} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_bmi} >= 25 ;
    XX_ML_test{id_con_1,id_bmi} = 1;
    XX_ML_test{~id_con_1,id_bmi} = 0;

    % ============== if Size
    id_size = indxFinder('Size');
    id_con_1 = XX_ML{:,id_size} >= 5;
    XX_ML{id_con_1,id_size} = 1;
    XX_ML{~id_con_1,id_size} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_size} >= 5 ;
    XX_ML_test{id_con_1,id_size} = 1;
    XX_ML_test{~id_con_1,id_size} = 0;
    

    % ============== if Age
    id_age = indxFinder('Age');
    id_con_1 = XX_ML{:,id_age} >= 65;
    XX_ML{id_con_1,id_age} = 1;
    XX_ML{~id_con_1,id_age} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_age} >= 65 ;
    XX_ML_test{id_con_1,id_age} = 1;
    XX_ML_test{~id_con_1,id_age} = 0;
    
    % ============== if Bilirubin
    id_bili = indxFinder('Bilirubin');
    id_con_1 = XX_ML{:,id_bili} > 20;
    XX_ML{id_con_1,id_bili} = 1;
    XX_ML{~id_con_1,id_bili} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_bili} > 20;
    XX_ML_test{id_con_1,id_bili} = 1;
    XX_ML_test{~id_con_1,id_bili} = 0;
    
    % ============== if INR
    id_inr = indxFinder('INR');
    id_con_1 = XX_ML{:,id_inr} > 1.1;
    XX_ML{id_con_1,id_inr} = 1;
    XX_ML{~id_con_1,id_inr} = 0;

    % Test
    id_con_1 = XX_ML_test{:,id_inr} > 1.1;
    XX_ML_test{id_con_1,id_inr} = 1;
    XX_ML_test{~id_con_1,id_inr} = 0;
    

    %==================================================
    

    %% tsne
%     y = tsne(XX_ML{:,2:end},'Algorithm','Exact','NumPCAComponents',10,statset('MaxIter',2000));
%     gscatter(y(:,1),y(:,2),XX_ML{:,1});

    %%
    X_raw_train = XX_ML{:,2:end}';
    X_raw_test = XX_ML_test{:,2:end}';

    Y_train = XX_ML{:,1};
    Y_test = XX_ML_test{:,1};
    
    %%
    for i = 1:100
        
        r_sample = randperm(size(X_raw_train,2));
        
        SampleSize = size(X_raw_train,2);
        
        Y_model = Y_train(r_sample);
        
        X_train = X_raw_train(:,r_sample);   
        r_features = randperm(size(X_raw_train,1));
        
        X_train = X_train(r_features,:);
        X_test = X_raw_test(r_features,:);
        %
        %% ML model
        x_train_test = X_train;
        Y_train_test = cellstr(num2str(Y_model));
        Y_test_china = cellstr(num2str(Y_test));
        
        Y_train_test(strcmp(Y_train_test,'1'))= {'Yes'};
        Y_train_test(strcmp(Y_train_test,'0'))= {'No'};

        Y_test_china(strcmp(Y_test_china,'1'))= {'Yes'};
        Y_test_china(strcmp(Y_test_china,'0'))= {'No'};
        
        cv = cvpartition(size(x_train_test,2),'HoldOut',0.2);
        idx = cv.test;
        
        
        [count,label] = hist(categorical(Y_train_test),unique(Y_train_test));
        %%
        XTrain = x_train_test;
        YTrain = Y_train_test;

        XTest = X_test;
        YTest = Y_test_china;

        %%
        % n = 1: RF, 2: KNN, 3: SVM
        
        % RF
        n = 1;
        
        disp(i);
        
        pred_test = [];

        [err,acc_train,acc,sen,spe,ppv,auc,order,cm,model,pred_train]= ...
            F_classifier_CV(XTrain',YTrain,n,true,{'Yes'});

        
        f1 = 2*(ppv*sen)/(ppv+sen);
        
        if ~isempty(auc)
            x_auc = auc.X_auc;
            y_auc = auc.Y_auc;
        else
            x_auc = [];
            y_auc = [];
        end


        %%
        words_row = XX_ML.Properties.VariableNames;
        
        Features = words_row(2:end);
        Features_order = Features(r_features);
        
        % If the model is Random Forest thereby the important predictor
        % factor can be calculated. 
        
        if n == 1
            
            imp = predictorImportance(model{1});
        
            [imp_sorted,index_imp]=sort(imp,'descend');
            
            Result(i,j).rankFeatures = imp_sorted;
            Result(i,j).FeaturesSorted = Features(index_imp);
            Result(i,j).indxe_imp = index_imp;
            
        end
%         
        
        Result(i,j).err = err;
        Result(i,j).acc = acc;
        Result(i,j).acc_train = acc_train;
        Result(i,j).sen = sen;
        Result(i,j).spe = spe;
        Result(i,j).f1 = f1;
        Result(i,j).auc = auc;
        Result(i,j).cm = cm;
        Result(i,j).model = model{1};
        Result(i,j).sampleOrder = r_sample;

        Result(i,j).ypred_train = pred_train;
        Result(i,j).ypred_test = pred_test;
        Result(i,j).ytest = YTest;
        Result(i,j).xtest = XTest;
        Result(i,j).ytrain = YTrain;
        Result(i,j).xtrain = XTrain;
        Result(i,j).xraw = X_raw_train;
  
        Result(i,j).Features = Features;
        Result(i,j).order = order;
        Result(i,j).idx = idx;
        Result(i,j).r_features = r_features;

        Result(i,j).Features_order = Features_order;

    end
    
end

%% Find the most important Features

imp = [];
Features_Imp = [];
fprintf('Please wait ... \n')
for j = 1:50
    for i = 1:length(Result)
        try
            Mdl = Result(i).model;
            Imp = oobPermutedPredictorImportance(Mdl);
            imp = [imp;Imp];
            Features_Imp = [Features_Imp;Result(i).Features_order];
            fprintf('=')
        catch
            continue
        end
        
    end
    disp(['# Evaluation: ' num2str(j)])
    if j == 50
        fprintf('\n Feature Importancy is done!\n')
    end

end
%% Make the order consistent for features
save('imp','imp');
save('Features_Imp','Features_Imp');

ref_order = sort(Features_Imp(1,:));
for i = 1:size(imp,1) %
    [~,id_f] = sort(Features_Imp(i,:));
    imp_sorted_features(i,:) = imp(i,id_f);
end
%% Plot the feature importancy
meanIMP = mean(imp_sorted_features);

[ccc,bb] = sort(meanIMP,'ascend');

MIN = min(imp_sorted_features);
MIN = MIN(bb);

MAX = max(imp_sorted_features);
MAX = MAX(bb);


Err = std(imp_sorted_features);
Err = Err(bb);
mod_features = strrep(ref_order,'_','.');

X = categorical(strrep(mod_features(bb),'_','.'));
X = reordercats(X,mod_features(bb));

for i = 1:length(X)
    
    if ccc(i)>0
        clr = 'r';
    else
        clr = 'b';
    end
    barh(X(i),ccc(i),clr)
    hold on
end

er = errorbar(ccc,X,Err/2,'.','horizontal');
er.LineWidth = 1.5;
er.Color = 'k';
er.MarkerSize = 1;

set(gca,'fontname','times')
xlabel('Importance Factor')
saveas(gcf,'ImportanceFigure.fig')
saveas(gcf,'ImportanceFigure.pdf')
%%
save('Final_100Runs_RF_SelecFeature','Result')



