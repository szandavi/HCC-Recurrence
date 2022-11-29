clc
clear
%% load best model
load('BestTrainedModel.mat')
%% Load Test Data
data_test = load('Data_test_after_binary.mat');
%%
data_test = data_test.data_test;
%% Features
id_features = @(x)(strcmp(x,data_test.Properties.VariableNames));
%
list_selected_features = {'INR','LiverDisease','No_Lesions',...
        'Ethnicity','Cirrhosis','DM','HPVG','ALT',...
        'eGFR','Albumin','AFP','BMI','Satellite',...
        'LVI','Sex','Bilirubin','Age','IHD','Size','PriorTACE'};

Features_RF_KM = list_selected_features;

T_X_ML_Sel = [];

for i = 1:length(Features_RF_KM)+1
    if i == 1
        indx = id_features('Recurrence');
    else
        indx = id_features(Features_RF_KM{i-1});
    end

    T_X_ML_Sel = [T_X_ML_Sel data_test(:,indx)];
end

XX_ML = T_X_ML_Sel;
%%
X_test_china = XX_ML{:,2:end};

X_test_china = X_test_china(:,r_features);
Y_china = XX_ML{:,1};

Y_test_china = cellstr(num2str(Y_china));
Y_test_china(strcmp(Y_test_china,'1'))= {'Yes'};
Y_test_china(strcmp(Y_test_china,'0'))= {'No'};


pre_test = net.test(X_test_china);

for i=1:size(pre_test)
    if pre_test(i,1) > pre_test(i,2)
        PreTest{i} = 'No';
        Y_pred_roc(i) = 0;
    else
        PreTest{i} = 'Yes';
        Y_pred_roc(i) = 1;
    end
end

cp_test = classperf(Y_test_china, PreTest,'Positive',{'Yes'},'Negative',{'No'});

acc =  cp_test.CorrectRate;
sen =  cp_test.Sensitivity;
spe =  cp_test.Specificity;
ppv = cp_test.PositivePredictiveValue;
[cm,order] = confusionmat(Y_test_china,PreTest);

f1 = 2*(ppv*sen)/(ppv+sen);


cm1 = confusionchart(cm,order, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
%     'OffDiagonalColor','red',...
%     'DiagonalColor','blue');

% sortClasses(cm,'descending-diagonal');

cm1.GridVisible = 'off';

cm1.FontName = 'Times New Roman';
cm1.Visible = 'on';

%%
acc = vertcat(Result.acc_train);
F1  = vertcat(Result.f1_train);
Sen = vertcat(Result.sen_train);
Spe = vertcat(Result.spe_train);

x_tick = {'Accuracy','F1 Score', 'Sensitivity', 'Specificity'};
x_boxplot = [acc F1 Sen Spe];


boxplot(x_boxplot,x_tick,'Notch','off','Widths',0.6)
% ylim([0.8 1])
set(findobj(gca,'type','line'),'linew',1.5)
set(gca,'fontname','times')

%%
figure;

[~,id_best] = max(vertcat(Result.f1_test));
cm = Result(id_best).cm_test;
order = Result(id_best).order_test;


cm1 = confusionchart(cm,order, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
%     'OffDiagonalColor','red',...
%     'DiagonalColor','blue');

% sortClasses(cm,'descending-diagonal');

cm1.GridVisible = 'off';

cm1.FontName = 'Times New Roman';
cm1.Visible = 'on';













