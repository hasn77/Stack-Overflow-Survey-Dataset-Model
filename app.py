import gradio as gr
import joblib
import pandas as pd
import numpy as np

from transformers import (
    AdvancedFeatureEngineer, 
    OrgSizeBinner, 
    CustomTargetEncoder, 
    bucketize_professional_tech
)


model = joblib.load('ensemble_model.joblib')
pipeline = joblib.load('salary_pipeline.joblib')

def predict_salary(years_pro, years_total, country, education, employment, 
                  dev_type, org_size, platform, webframe, prof_tech, 
                  languages, tools, age_group):
    """
    Predicts salary using the loaded ML pipeline and model.
    """

    input_data = pd.DataFrame({
        'YearsCodePro_B': [years_pro],
        'YearsCode': [years_total],
        'Country': [country],
        'EdLevel_Bucket': [education],
        'Employment_Category_Bucket': [employment],
        'DevType_Bucket': [dev_type],
        'OrgSize': [org_size],
        'PlatformHaveWorkedWith_Bucket': [platform],
        'WebframeHaveWorkedWith_Bucket': [webframe],
        'ProfessionalTech': [prof_tech], 
        'LanguageHaveWorkedWith_Bucket': [languages],
        'ToolsTechHaveWorkedWith_Bucket': [tools],
        'Age_Encoded': [age_group]
    })

    # --- 4. APPLY the transformations ---
    # Apply the custom transformers to the input data
    input_data['ProfessionalTech_Bucket'] = input_data['ProfessionalTech'].apply(bucketize_professional_tech)
    
    top_10_countries = ['United States', 'Germany', 'United Kingdom', 'Canada', 'India', 
                       'France', 'Netherlands', 'Australia', 'Poland', 'Brazil']
    input_data['Country_Grouped'] = input_data['Country'].apply(
        lambda x: x if x in top_10_countries else 'Other'
    )

    transformed_data = pipeline.transform(input_data)
    predicted_salary_log = model.predict(transformed_data)
    

    predicted_salary = np.expm1(predicted_salary_log)
    

    return f"${predicted_salary[0]:,.2f}"


def create_interface():
    pass

if __name__ == "__main__":
    demo = create_interface() 
    demo.launch()