import gradio as gr
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Custom Transformers - Exact copies from your pipeline
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, salaries, y=None):
        self.feature_names_in = salaries.columns.tolist()
        return self
    
    def transform(self, salaries):
        salaries = salaries.copy()
        
        # Convert to numeric
        salaries['YearsCode'] = pd.to_numeric(salaries['YearsCode'], errors='coerce').fillna(0)
        salaries['YearsCodePro_B'] = pd.to_numeric(salaries['YearsCodePro_B'], errors='coerce').fillna(0)
        
        # Feature engineering logic
        skill_columns = [col for col in salaries.columns if col.endswith('_Bucket')]
        
        salaries['skill_diversity'] = salaries[skill_columns].apply(
            lambda row: len([val for val in row if val not in ['None', 'Other', 'Not Specified']]), axis=1
        ).fillna(0)
        
        salaries['experience_consistency'] = salaries['YearsCodePro_B'] / (salaries['YearsCode'] + 1)
        salaries['experience_consistency'] = salaries['experience_consistency'].clip(0, 1).fillna(0)
        
        seniority_keywords = ['Senior', 'Lead', 'Staff', 'Principal', 'Manager', 'Director']
        salaries['is_senior_role'] = salaries['DevType_Bucket'].str.contains('|'.join(seniority_keywords), case=False, na=False)
        
        salaries['professional_experience_factor'] = (
            salaries['YearsCodePro_B'] * 0.7 + 
            salaries['experience_consistency'] * 0.2 + 
            salaries['is_senior_role'].astype(int) * 0.1
        )
        
        salaries['experience_skill_ratio'] = salaries['skill_diversity'] / (salaries['YearsCodePro_B'] + 1)
        
        salaries['senior_experience_match'] = (
            salaries['is_senior_role'] & (salaries['YearsCodePro_B'] >= 5)
        ).astype(int)
        
        return salaries

class OrgSizeBinner(BaseEstimator, TransformerMixin):
    def fit(self, salaries, y=None):
        return self
    
    def transform(self, salaries):
        salaries = salaries.copy()
        
        orgsize_bins = {
            'large_enterprise': ['10,000 or more employees'],
            'enterprise': ['5,000 to 9,999 employees'], 
            'mid_company': ['1,000 to 4,999 employees'],
            'small_company': ['500 to 999 employees', '100 to 499 employees'],
            'startup': ['20 to 99 employees', '2 to 19 employees'],
            'freelancer': ['Just me - I am a freelancer, sole proprietor, etc.'],
            'other': ['I don\\t know', 'Not specified']
        }
        
        salaries['OrgSize_Binned'] = 'other'  # default
        for bin_name, categories in orgsize_bins.items():
            salaries.loc[salaries['OrgSize'].isin(categories), 'OrgSize_Binned'] = bin_name
            
        return salaries

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target encoder requires y during fit")

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")
        else:   
            y = y.copy()

        self.global_mean_ = y.mean()
        self.encodings_ = {}
        
        for col in X.columns:
            salary = pd.concat([X[[col]], y], axis=1)
            self.encodings_[col] = salary.groupby(col)[y.name].mean()
            
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in X.columns:
            X_new[col] = X_new[col].map(self.encodings_[col]).fillna(self.global_mean_)
        return X_new

def bucketize_professional_tech(tech_string):
    """Professional tech bucketization"""
    if pd.isna(tech_string):
        return 'None'
    
    lower_tech_string = str(tech_string).lower()

    if 'none of these' in lower_tech_string:
        return 'None'

    buckets = {
        'AI/ML': ['ai', 'machine learning', 'ml', 'deep learning', 'neural network', 'nlp', 'natural language', 'computer vision', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'openai'],
        'Data Science & Analytics': ['data science', 'data analysis', 'analytics', 'big data', 'hadoop', 'spark', 'pandas', 'numpy', 'tableau', 'power bi', 'databricks', 'snowflake'],
        'DevOps & Cloud': ['devops', 'ci/cd', 'continuous integration', 'continuous delivery', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'aws', 'azure', 'gcp', 'cloud', 'observability'],
        'Web Development': ['web', 'frontend', 'backend', 'full-stack', 'javascript', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'ruby on rails', 'php', 'asp.net'],
        'Mobile Development': ['mobile', 'ios', 'android', 'swift', 'kotlin', 'react native', 'flutter', 'xamarin'],
        'Databases': ['database', 'sql', 'nosql', 'postgresql', 'mysql', 'sql server', 'mongodb', 'redis', 'cassandra', 'firebase'],
        'Testing & QA': ['testing', 'qa', 'quality assurance', 'selenium', 'jest', 'pytest', 'cypress', 'junit'],
        'Security': ['security', 'cybersecurity', 'infosec', 'penetration testing', 'pen testing'],
        'Developer Tools': ['git', 'github', 'gitlab', 'jira', 'visual studio code', 'ide'],
        'Architecture & Practices': ['microservices', 'developer portal', 'innersource']
    }

    listed_techs = [tech.strip().lower() for tech in str(tech_string).split(';')]

    for tech in listed_techs:
        for bucket, keywords in buckets.items():
            if any(keyword in tech for keyword in keywords):
                return bucket

    return 'Other'

# Load your trained model and pipeline (you'll upload these files)
# ensemble_model = pickle.load(open('salary_ensemble_model.pkl', 'rb'))
# preprocessing_pipeline = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))

def predict_salary(years_pro, years_total, country, education, employment, 
                  dev_type, org_size, platform, webframe, prof_tech, 
                  languages, tools, age_group):
    """
    Complete salary prediction using your exact pipeline
    """
    # For demo purposes - replace with actual model loading
    # This is a placeholder that mimics your feature structure
    
    # Create input dataframe matching your pipeline expectations
    input_data = pd.DataFrame({
        'YearsCodePro_B': [years_pro],
        'YearsCode': [years_total],
        'Country': [country],
        'EdLevel_Bucket': [education],
        'Employment_Category_Bucket': [employment],
        'DevType_Bucket': [dev_type],
        'OrgSize': [org_size],  # Raw categorical for OrgSizeBinner
        'PlatformHaveWorkedWith_Bucket': [platform],
        'WebframeHaveWorkedWith_Bucket': [webframe],
        'ProfessionalTech': [prof_tech],  # Raw for bucketization
        'LanguageHaveWorkedWith_Bucket': [languages],
        'ToolsTechHaveWorkedWith_Bucket': [tools],
        'Age_Encoded': [age_group]
    })
    
    # Apply professional tech bucketization
    input_data['ProfessionalTech_Bucket'] = input_data['ProfessionalTech'].apply(bucketize_professional_tech)
    
    # Apply country grouping (simplified for demo)
    top_10_countries = ['United States', 'Germany', 'United Kingdom', 'Canada', 'India', 
                       'France', 'Netherlands', 'Australia', 'Poland', 'Brazil']
    input_data['Country_Grouped'] = input_data['Country'].apply(
        lambda x: x if x in top_10_countries else 'Other'
    )
    
    # Placeholder prediction (replace with actual pipeline.predict())
    base_salary = 75000
    experience_factor = years_pro * 3000
    country_factor = 25000 if country == "United States" else 10000 if country in ["Germany", "United Kingdom", "Canada"] else 0
    
    predicted_salary = base_salary + experience_factor + country_factor
    
    return f"${predicted_salary:,}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Tech Salary Predictor", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 💰 Tech Salary Predictor")
        gr.Markdown("**The $432K → $52K RMSE Journey** | Stack Overflow 2023 Survey | Final RMSE: $52,569")
        
        with gr.Row():
            with gr.Column():
                years_pro = gr.Slider(
                    minimum=0, maximum=30, value=3, step=1,
                    label="Years of Professional Coding Experience (capped at 30)"
                )
                
                years_total = gr.Slider(
                    minimum=0, maximum=50, value=5, step=1,
                    label="Total Years Coding (including hobbyist)"
                )
                
                country = gr.Dropdown(
                    choices=["United States", "Germany", "United Kingdom", "Canada", 
                            "India", "France", "Netherlands", "Australia", "Poland", "Brazil", "Other"],
                    value="United States",
                    label="Country (Top 10 + Other)"
                )
                
                education = gr.Dropdown(
                    choices=["Masters", "Bachelors", "No_Degree", "Associates"],
                    value="Bachelors",
                    label="Education Level (Ordinal Encoded)"
                )
                
                employment = gr.Dropdown(
                    choices=["Full-time Employed", "Student", "Freelancer/Self-employed", 
                            "Part-time Employed", "Other"],
                    value="Full-time Employed",
                    label="Employment Category"
                )
                
                age_group = gr.Dropdown(
                    choices=["1", "2", "3", "4", "5", "6"],  # Age encoded as integers
                    value="3",
                    label="Age Group (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55-64, 6=65+)"
                )
            
            with gr.Column():
                dev_type = gr.Dropdown(
                    choices=["Full-stack developer", "Back-end developer", "Front-end developer", 
                            "Senior Executive", "DevOps specialist", "Data scientist", 
                            "Engineering manager", "Other"],  # Placeholder - you'll fill these
                    value="Full-stack developer",
                    label="Developer Type (Target Encoded)"
                )
                
                org_size = gr.Dropdown(
                    choices=["10,000 or more employees", "5,000 to 9,999 employees", 
                            "1,000 to 4,999 employees", "500 to 999 employees", 
                            "100 to 499 employees", "20 to 99 employees", 
                            "2 to 19 employees", "Just me - I am a freelancer, sole proprietor, etc.",
                            "I don't know", "Not specified"],
                    value="100 to 499 employees",
                    label="Organization Size (Raw - gets binned by pipeline)"
                )
                
                platform = gr.Dropdown(
                    choices=["AWS", "Google Cloud Platform", "Microsoft Azure", "Docker", 
                            "Kubernetes", "Firebase", "Other", "None"],
                    value="AWS",
                    label="Platform (Target Encoded)"
                )
                
                webframe = gr.Dropdown(
                    choices=["React.js", "Node.js", "Vue.js", "Angular", "Django", 
                            "Flask", "Spring", "Other", "None"],
                    value="React.js",
                    label="Web Framework (Target Encoded)"
                )
                
                languages = gr.Dropdown(
                    choices=["Python", "JavaScript", "Java", "C#", "TypeScript", 
                            "Go", "Rust", "C++", "Other"],
                    value="Python",
                    label="Primary Language (Target Encoded)"
                )
                
                tools = gr.Dropdown(
                    choices=["Git", "Docker", "Kubernetes", "Jenkins", "npm", 
                            "Yarn", "Webpack", "Other", "None"],
                    value="Git", 
                    label="Tools/Tech (Target Encoded)"
                )
                
                prof_tech = gr.Textbox(
                    value="Machine learning;Data analysis",
                    label="Professional Tech (semicolon separated - gets auto-bucketized)",
                    placeholder="e.g., Machine learning;Data analysis;Docker;AWS"
                )
        
        with gr.Row():
            predict_btn = gr.Button("🚀 Predict Salary (Ensemble Model)", variant="primary", size="lg")
            
        with gr.Row():
            output = gr.Textbox(
                label="Predicted Annual Salary (USD)", 
                placeholder="Ready to predict your tech salary...",
                text_align="center"
            )
        
        # Connect the button to the prediction function
        predict_btn.click(
            fn=predict_salary,
            inputs=[years_pro, years_total, country, education, employment,
                   dev_type, org_size, platform, webframe, prof_tech,
                   languages, tools, age_group],
            outputs=output
        )
        
        gr.Markdown("""
        ### 🎯 **The Journey: From Disaster to Victory**
        
        **📈 Performance Evolution:**
        - **Initial Disaster:** $432,861 RMSE (extreme outliers)
        - **Data Cleaning:** ~$54K RMSE (realistic salary range)  
        - **Feature Engineering:** $53,347 RMSE (6 custom features)
        - **Advanced Models:** $52,951 RMSE (Gradient Boosting)
        - **Bug Fixes:** $52,647 RMSE (OrgSize double-encoding fix)
        - **🏆 Final Ensemble:** $52,569 RMSE (**88% improvement!**)
        
        **🔧 Technical Highlights:**
        - **Custom Transformers:** OrgSizeBinner, AdvancedFeatureEngineer, CustomTargetEncoder
        - **Ensemble Learning:** Random Forest + Gradient Boosting + XGBoost
        - **Smart Feature Engineering:** 6 domain-driven features (experience consistency, skill diversity, etc.)
        - **Robust Pipeline:** Leak-proof train/test separation with proper data flow
        
        **🐛 Major Debugging Victories:**
        - The `random_state=42` saga (data integrity over model complexity)
        - OrgSize double-encoding bug fix (27.8% → realistic feature importance)
        - Pipeline integration issues (feature engineering in transformers)
        
        **📊 Key Salary Drivers:**
        1. **Organization Size (27.8%)** - Company size matters most
        2. **Professional Experience (22.7%)** - Years of expertise  
        3. **Programming Languages (17.1%)** - Tech stack impact
        4. **Web Frameworks (6.8%)** - Specialization premium
        5. **Developer Type (6.5%)** - Role seniority effect
        
        ### ⚠️ Disclaimer
        Educational ML project showcasing end-to-end pipeline development. Predictions are estimates based on 2023 Stack Overflow survey data.
        
        *"The answer to life, the universe, and everything is 42... and sometimes it's also the random_state that saves your ML project!"* 🚀
        """)
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()