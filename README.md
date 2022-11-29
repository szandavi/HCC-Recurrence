# Pipeline

![Pipeline](./Pipeline.png)

# Background
Artificial intelligence (AI) has recently gathered significant interest in the field of predictive medicine. We aimed to apply AI to comprehensive clinical datasets to develop and validate an accurate AI model for predicting the risk of HCC recurrence 1-year post-surgical resection. 

# Methods
It is a retrospective multi-centre study that included patients who underwent surgical resection in three centres in Australia and one in Hong Kong between 2000 – 2019. The Australian and Hong Kong cohorts had major differences due to the underlying liver disease. Thus, model development/training comprised the cohorts collected from Australian centres, plus approximately 60% of patient samples randomly picked from the Hong Kong cohort (excluding patients with non-alcoholic fatty liver disease (NAFLD)) to ensure sufficient heterogeneity in the discovery dataset and, therefore, model generalisability. The remaining patients from the Hong Kong cohort were held out for next-level independent (quasi-external) validation. We used multiple metrics to measure the prediction performance on internal and quasi-external validation sets.  

# Findings
A total of 664 patients with greater than 365 days of follow-up after surgical resection were included in the model. The development cohort consisted of 466 patients: 167 patients from three liver centres in Australia and 299 patients from the Chinese University of Hong Kong (CUHK), whilst the external validation cohort consisted of 198 patients from CUHK. AI algorithms identified variables that co-act to positively contribute to the prediction of recurrence, profiling high-risk patients to be males, with underlying significant liver disease due to alcohol or chronic hepatitis C, co-existing with type II diabetes mellitus, increased body mass index, and multiple metabolic complications such as NAFLD, chronic kidney disease, ischaemic heart disease and cerebrovascular disease. For internal validation, the model achieved the following performance metrics: accuracy = 0.946±0.005 (mean ± standard deviation), sensitivity = 0.938±0.014, specificity = 0.954±0.0165, and AUC = 0.853±0.013. We also performed a patient-level assessment of the model’s predictions across 100 iterations. We observed that 94% of patients were correctly classified in at least 75% of runs, indicating the power of the model to predict the status of the recurrence for every single patient. The same model was able to predict recurrence for the 198 patients recruited from CUHK with predictive performances of accuracy = 0.864, sensitivity = 0.870, and specificity = 0.856, further demonstrating the model’s generalisability.

# Interpretation 
AI models reliably identified a personalised profile combining a combination of variables that act synergistically to increase HCC recurrence risk. Notably, some of the identified risk factors, such as metabolic risk factors, alcohol intake, or even the underlying aetiology of liver disease, are potentially reversible or modifiable, hence inviting interventions to disrupt the network supporting hepatocarcinogenesis. Therefore, AI predictive models have the potential to significantly benefit our clinical practice for personalised risk stratification and mitigation. 

# Code:
The code is implemented in MATLAB R2020b, and the saved results can be found in the shared [Link](https://drive.google.com/file/d/1dhKK-0V2vRhIaFg137MwqsDUFUMlJcK9/view?usp=sharing). Note that it is required to download *Final_RF_10Fold_365.mat* and *Final_SVM_10Fold_SelectedFeatures.mat* from the shared [Link](https://drive.google.com/file/d/1dhKK-0V2vRhIaFg137MwqsDUFUMlJcK9/view?usp=sharing) if you have a plan to run *Result.m* without running *main.m* and *main_selectedFeatures.m*. 


# Reference: 
 
