# Background: 
Post resection of hepatocellular carcinoma (HCC), around 70% of patients experience tumour recurrence at 5 years post-surgery.  Studies regarding factors
predictive of HCC recurrence post-surgical resection have yielded differing, and sometimes conflicting data. Artificial intelligence (AI) has recently gathered
significant interests in the field of predictive medicine driven by increases in computational power and a rising number of multidimensional patient data
available to predict clinical outcomes. AI algorithms allow individualised patient profiling, which is essential for patient stratification, and characterisation
of individual risk of developing a disease. We aimed to apply AI to a comprehensive clinical dataset to identify at the patient-level, predictive risks of HCC
recurrence occurring 1-year post surgery. 
# Methods: 
This was a retrospective study examining a comprehensive dataset on patients who underwent surgical resection in 2 liver centres, in the period between 2000-
2019. Extensive multidimensional detailed dataset pertaining to patient, tumour and liver disease parameters were collected. Patient data included co morbidities
such as ischaemic heart disease, cerebrovascular disease, chronic kidney disease and metabolic risk factors. To develop the AI model, we followed best
practices for AI predictive modelling by developing different classifiers (for performance comparison) as well as extensive validations. Performance of the
AI model was tested by receiver operating characteristic (ROC) curve, and the confusion matrix which details the total number of correct and incorrect
predictions for each patient. 
# Results: 
A total of 134 patients who had HCC recurrence 1-year post-surgical resection patients were included in the study. The AI generated model profiled patients who
are at risk of developing post-surgical HCC recurrence, to be of male gender, with underlying significant liver disease due to alcohol or non-alcohol related
liver disease, of increased body mass index, with multiple metabolic risk factors and related complications such as, non-alcoholic fatty liver disease, chronic
kidney disease and cerebrovascular disease. The AI model achieved a high predictive performance, upon exhaustive validation, as measured by multiple metrics with
accuracy of 82%, sensitivity of 83%, specificity of 81%, and AUC of 83%. A patient-level assessment of the model’s predictions was performed to investigate
whether some patients are frequently miss-classified by the model across 500 iterations. This confirmed that all patients were correctly classified in 83% to 91%
of runs indicating the power of the model to predict the status of the recurrence for every single patient.
# Conclusions: 
We reliably identified through AI a personalised patient’s clinical profile combining host, tumour and liver related factors collectively contributing to the
prediction of HCC recurrence after surgical resection. Notably, some of the identified risk factors such as BMI, metabolic risk factors, alcohol intake, or
even the extent of the underlying liver disease are potentially reversible or modifiable.
# Interpretation: 
Unlike conventional statistical methods, AI can learn complex patterns and accordingly consider all variables that could potentially be dismissed by
conventional statistics if they do not meet the set threshold value (P<0.05). In this study, AI algorithm reliably allowed identification of individualised
patient profiling variables that act synergistically to increase the risk of HCC recurrence post-surgical resection.  This approach is important in clinical
practice as it can guide developing personalised preventative strategies to reduce the risk of HCC recurrence. 

# Code:
The code is implemented in MATLAB R2020b, and the saved results can be found in the shared [link](https://drive.google.com/file/d/1dhKK-0V2vRhIaFg137MwqsDUFUMlJcK9/view?usp=sharing).


# Reference: 
 
