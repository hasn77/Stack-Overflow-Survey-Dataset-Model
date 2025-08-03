import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"


try:
    from transformers import (
        AdvancedFeatureEngineer, 
        OrgSizeBinner, 
        CustomTargetEncoder,
        bucketize_professional_tech
    )
except ImportError:

    def bucketize_professional_tech(tech):
        return tech


try:
    model = joblib.load('ensemble_model.joblib')
    pipeline = joblib.load('pipeline.joblib')
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Model loading failed: {e}")
    model, pipeline = None, None


def predict_salary(years_pro, dev_type, education, languages, prof_tech, tools, 
                  employment, country, org_size, platform, age_group):
    
    if model is None or pipeline is None:
        return "❌ ERROR: Models not loaded properly"

    try:
        dev_type_str = str(dev_type)
        languages_str = str(languages)
        prof_tech_str = str(prof_tech)
        tools_str = str(tools)
        platform_str = str(platform)
        

        years_total = max(years_pro + 2, years_pro)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'YearsCodePro_B': [years_pro],
            'YearsCode': [years_total],
            'Country': [country],
            'EdLevel_Bucket': [education],
            'Employment_Category_Bucket': [employment],
            'DevType_Bucket': [dev_type_str],
            'OrgSize': [org_size],
            'PlatformHaveWorkedWith_Bucket': [platform_str],
            'WebframeHaveWorkedWith_Bucket': ['None'],  
            'ProfessionalTech': [prof_tech_str], 
            'LanguageHaveWorkedWith_Bucket': [languages_str],
            'ToolsTechHaveWorkedWith_Bucket': [tools_str],
            'Age_Encoded': [int(age_group)]  
        })

        input_data['ProfessionalTech_Bucket'] = input_data['ProfessionalTech'].apply(bucketize_professional_tech)
        

        top_10_countries = ['United States', 'Germany', 'United Kingdom', 'Canada', 'India', 
                           'France', 'Netherlands', 'Australia', 'Poland', 'Brazil']
        input_data['Country_Grouped'] = input_data['Country'].apply(
            lambda x: x if x in top_10_countries else 'Other'
        )


        transformed_data = pipeline.transform(input_data)
        predicted_salary_log = model.predict(transformed_data)
        predicted_salary = np.expm1(predicted_salary_log[0])
        
        return f"💰 ${predicted_salary:,.0f}"
        
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"


years_pro = gr.Slider(
    label="📈 Years of Professional Experience", 
    minimum=0, maximum=30, value=5, step=1
)

dev_type = gr.Dropdown(
    label="💻 Developer Type",
    choices=['Full Stack Developer', 'Backend Developer', 'Frontend Developer', 
            'Data Scientist', 'DevOps Engineer', 'Mobile Developer', 'Other'],
    value='Full Stack Developer'
)

education = gr.Dropdown(
    label="🎓 Education Level",
    choices=['Masters', 'Bachelors', 'No_Degree', 'Associates'],
    value='Bachelors'
)

languages = gr.Dropdown(
    label="🐍 Primary Programming Language",
    choices=['Python', 'JavaScript', 'Java', 'C#', 'TypeScript', 'Go', 'Other'],
    value='Python'
)

prof_tech = gr.Dropdown(
    label="🚀 Professional Tech Area",
    choices=['Web Development', 'AI/ML', 'Data Science & Analytics', 
            'DevOps & Cloud', 'Mobile Development', 'Other'],
    value='Web Development'
)

tools = gr.Dropdown(
    label="🛠️ Primary Tools",
    choices=['Git', 'Docker', 'Kubernetes', 'CI/CD Tools', 'Other'],
    value='Git'
)

employment = gr.Dropdown(
    label="💼 Employment Type",
    choices=['Full-time Employed', 'Student', 'Freelancer/Self-employed', 
            'Part-time Employed', 'Other'],
    value='Full-time Employed'
)

country = gr.Dropdown(
    label="🌍 Country",
    choices=['United States', 'Germany', 'United Kingdom', 'Canada', 
            'India', 'France', 'Netherlands', 'Australia', 'Poland', 
            'Brazil', 'Other'],
    value='United States'
)

org_size = gr.Dropdown(
    label="🏢 Organization Size",
    choices=['10,000 or more employees', '5,000 to 9,999 employees',
            '1,000 to 4,999 employees', '500 to 999 employees',
            '100 to 499 employees', '20 to 99 employees',
            '2 to 19 employees', 'Just me - I am a freelancer, sole proprietor, etc.',
            'I dont know', 'Not specified'],
    value='100 to 499 employees'
)

platform = gr.Dropdown(
    label="☁️ Cloud Platform",
    choices=['Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud', 
            'Firebase', 'Other', 'None'],
    value='Amazon Web Services (AWS)'
)

age_group = gr.Dropdown(
    label="👤 Age Group",
    choices=['1', '2', '3', '4', '5', '6'],
    value='3'
)

output = gr.Textbox(
    label="💰 Predicted Annual Salary (USD)",
    placeholder="Click predict to see your estimated salary..."
)

# Create the interface
demo = gr.Interface(
    fn=predict_salary,
    inputs=[years_pro, dev_type, education, languages, prof_tech, tools,
           employment, country, org_size, platform, age_group],
    outputs=output,
    title="💰 Tech Salary Predictor",
    description="Stack Overflow Survey 2023 Data",
    theme=gr.themes.Glass(),
    flagging_mode="never"
)

# --- 4. LAUNCH ---
if __name__ == "__main__":
    demo.launch()