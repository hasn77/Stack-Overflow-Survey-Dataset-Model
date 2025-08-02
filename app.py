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


def predict_salary(years_pro, dev_type, education, tools, prof_tech, languages, employment, 
                   country, org_size, platform, age_group, 
                   years_total=None, webframe=None):
    
    dev_type = ';'.join(dev_type) if isinstance(dev_type, list) else dev_type
    tools = ';'.join(tools) if isinstance(tools, list) else tools
    prof_tech = ';'.join(prof_tech) if isinstance(prof_tech, list) else prof_tech
    languages = ';'.join(languages) if isinstance(languages, list) else languages
    platform = ';'.join(platform) if isinstance(platform, list) else platform

    if years_total is None:
        years_total = years_pro + 2 if years_pro is not None else 2

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
    
    with gr.Blocks() as demo:
        gr.Markdown("## Tech Salary Prediction Interface")

        with gr.Row():
            with gr.Column():

                # Input fields for user data = Years Pro, DevType, Education, Tools, Professional Tech, Languages, Employment Category

                yearspro = gr.Slider(label="Years of Professional Coding Experience", minimum=0, maximum=10, step=1, value=None)

                devtype = gr.Dropdown(label='Developer Type', choices=['Frontend Developer', 'Backend Developer', 'Full Stack Developer',
                                                             'Data Scientist', 'DevOps Engineer', 'Mobile Developer',
                                                             'Game Developer', 'Other'], multiselect=True, value=None)

                education = gr.Dropdown(label='Education Level', choices=['PhD', 'Master/s Degree', 'Bachelor/s Degree',
                                                              'Associate/s Degree', 'No Degree'], value=None)

                primary_tool = gr.Dropdown(label='Primary Development Tool', choices=['Kubernetes', 'Docker', 'Cloud Platforms', 'Terraform', 'Ansible', 'Apache Spark', 'Apache Kafka',
                                                                       'AI/ML Libraries', 'Snowflake', 'PostgreSQL', 'MongoDB', 'Redis', 'CI/CD Tools', 'Game Engines',
                                                                       'Package Managers', 'Build Tools', 'Developer Tools'], multiselect=True, value=None)

                key_prof_tech = gr.Dropdown(label='Key Professional Technologies', choices=['AI/ML', 'Data Science/Analytics', 'Web Development', 'DevOps & Cloud Infrastructure',
                                                                            'Mobile Development', 'Databases', 'Testing & QA', 'Security',
                                                                            'Developer Tools'], multiselect=True, value=None)

                prog_lang = gr.Dropdown(label='Programming Language Proficient In', choices=['Python', 'JavaScript', 'Java', 'C#', 'C++', 'PHP', 'Ruby', 'Go',
                                                                                 'Swift', 'Kotlin', 'Rust', 'TypeScript'], multiselect=True, value=None)

                employment_type = gr.Dropdown(label='Employment Type', choices=['Full-time', 'Part-time', 'Contract', 'Freelance', 'Internship'], value=None)

                output_salary = gr.Textbox(label="Predicted Annual Salary (USD)")

                predict_btn = gr.Button("Predict Salary", variant="Primary")

        with gr.Accordion("Advanced Options (For detailed prediction)", open=False):
                gr.Markdown("Select additional options to refine your prediction.")

                country = gr.Dropdown(label='Country', choices=['United States', 'Germany', 'United Kingdom', 'Canada', 'India', 
                       'France', 'Netherlands', 'Australia', 'Poland', 'Brazil'], value=None)

                org_size = gr.Dropdown(label='Organization Size', choices=['20-99 employees', '100-499 employees', '1000 to 4,999 employees', '5,000 to 9,999 employees',
                                                                           '10,000 or more employees'], value=None)
                
                platforms = gr.Dropdown(label='Platforms', choices=['Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud',
                                                                    'Firebase', 'Cloudfare', 'DigitalOcean', 'Heroku', 'VMware',
                                                                    'Hetzner', 'Netlify', 'Managed Hosting', 'Vercel', 
                                                                    'Linode, now Akamai', 'OpenShift', 'OVH', 'Fly.io'
                                                                    'OpenStack', 'Colocation', 'Vultr'], multiselect=True, value=None)
                
                age_group = gr.Dropdown(label='Age Group', choices=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], value=None)

                predict_btn.click(
                    fn=predict_salary,
                    inputs=[yearspro, devtype, education, primary_tool, key_prof_tech, prog_lang, employment_type, country, org_size, platforms, age_group],
                    outputs=output_salary
                )

if __name__ == "__main__":
    demo = create_interface() 
    demo.launch()