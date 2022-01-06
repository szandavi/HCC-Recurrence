clc;
clear;
%% Load RF's result using all features over 10 folds cross validation
load('Final_RF_10Fold_365.mat');
%%
imp = [];
Features_Imp = [];
fprintf('Please wait ... \n')
for j = 1:50
    for i = 1:length(Result)
        try
            Mdl = Result(i).model;
            Imp = oobPermutedPredictorImportance(Mdl);%,'Options',...
%                                                statset('UseParallel',true));
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
%%
acc = vertcat(Result.acc);
F1  = vertcat(Result.f1);
Sen = vertcat(Result.sen);
Spe = vertcat(Result.spe);

x_tick = {'Accuracy','F1 Score', 'Sensitivity', 'Specificity'};
x_boxplot = [acc F1 Sen Spe];

boxplot(x_boxplot,x_tick,'Notch','off','Widths',0.6)
ylim([0.4 0.95])
set(findobj(gca,'type','line'),'linew',1.5)
set(gca,'fontname','times')

title('Performance of RF based on 10 Folds cross-validation for All Features over 500 executions')
%% Boxplot of 500 execusions for SVM on selected Features.
clear Result
load('Final_SVM_10Fold_SelectedFeatures')
%%
acc = vertcat(Result.acc);
F1  = vertcat(Result.f1);
Sen = vertcat(Result.sen);
Spe = vertcat(Result.spe);

x_tick = {'Accuracy','F1 Score', 'Sensitivity', 'Specificity'};
x_boxplot = [acc F1 Sen Spe];

boxplot(x_boxplot,x_tick,'Notch','off','Widths',0.6)
ylim([0.4 0.95])
set(findobj(gca,'type','line'),'linew',1.5)
set(gca,'fontname','times')

title('Performance of SVM based on 10 Folds cross-validation for Selected Features over 500 executions')
%% AUC
auc = [];
Y_auc = [];
X_auc = [];
[~,id_best] = max(vertcat(Result.acc));

for i = 1:size(Result,1)
    AUC = Result(i).auc;
    
    x_auc = AUC.X_auc;
    
    y_auc = AUC.Y_auc;
        
    auc = [auc;AUC.auc];
    
    plot(x_auc,y_auc,'Color',[0 1 1])
    hold on
    
end

% mean plot
plot (Result(id_best).auc.X_auc,Result(id_best).auc.Y_auc,'r','LineWidth',2)

hold on
plot([0 1],[0 1],'--k','LineWidth',2)

xlabel('1-Specificity')
ylabel('Sensitivity')

set(gca,'fontname','times')

%% Confusion Matrix
figure;

[~,id_best] = max(vertcat(Result.acc));
cm = Result(id_best).cm;
order = Result(id_best).order;

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
