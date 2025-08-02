import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
            'other': ['I don\'t know', 'Not specified']
        }
        
        salaries['OrgSize_Binned'] = 'other'  # default
        for bin_name, categories in orgsize_bins.items():
            salaries.loc[salaries['OrgSize'].isin(categories), 'OrgSize_Binned'] = bin_name
        return salaries


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, salaries, y=None):
        self.feature_names_in_ = salaries.columns.tolist()
        return self
    
    def transform(self, salaries):
        salaries = salaries.copy()
        
        salaries['YearsCode'] = pd.to_numeric(salaries['YearsCode'], errors='coerce').fillna(0)
        salaries['YearsCodePro_B'] = pd.to_numeric(salaries['YearsCodePro_B'], errors='coerce').fillna(0)
        
        skill_columns = [col for col in salaries.columns if col.endswith('_Bucket')]
        
        # Bug fixed here: used 'salaries' instead of 'X'
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
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None: 
            input_features = getattr(self, 'feature_names_in_', [])
        if hasattr(input_features, 'tolist'):
            input_features = input_features.tolist()
        engineered_features = [
            'skill_diversity', 'experience_consistency', 'is_senior_role',
            'professional_experience_factor', 'experience_skill_ratio', 'senior_experience_match'
        ]
        return input_features + engineered_features


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
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return input_features