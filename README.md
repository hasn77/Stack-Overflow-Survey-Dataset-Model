# Stack Overflow Developer Salary Prediction

A complete end-to-end machine learning project predicting developer salaries using the Stack Overflow 2023 Survey dataset. Built as part of mastering Chapter 2 concepts from "Hands-On Machine Learning" by Aurélien Géron.

## 🎯 Project Overview

**Goal**: Predict yearly developer salaries (ConvertedCompYearly) using survey responses  
**Dataset**: Stack Overflow 2023 Survey (~48K salary records)  
**Final Performance**: **$52,569 RMSE** using ensemble methods

## 🚀 Key Results

- **88% Performance Improvement**: From $432K RMSE (initial disaster) to $52K RMSE (final ensemble)
- **Advanced Pipeline**: Custom transformers, target encoding, and leak-proof preprocessing
- **Ensemble Victory**: VotingRegressor combining Random Forest, and Gradient Boosting

## 🔧 Technical Architecture

### Data Preprocessing Pipeline
- **Multi-label Categorical Handling**: Processed semicolon-separated survey responses (languages, platforms, tools)
- **Strategic Bucketization**: Reduced high-cardinality features using domain knowledge
- **Custom Transformers**: Built `AdvancedFeatureEngineer` and `OrgSizeBinner` for pipeline integration

### Feature Engineering
- **Experience Metrics**: Consistency ratios, professional experience factors, senior role indicators
- **Salary-Proportional Encoding**: Target encoding for tech stack features based on actual salary impact
- **Outlier Management**: Capped salaries ($50K-$750K) and experience years (max 30) for realistic predictions

### Model Development
- **Baseline Models**: Linear Regression, Random Forest establishing $53-54K RMSE baseline
- **Advanced Algorithms**: Gradient Boosting, XGBoost, Extra Trees, Ridge, ElasticNet, SVR
- **Hyperparameter Optimization**: GridSearchCV with 108 parameter combinations across 3-fold CV
- **Final Ensemble**: VotingRegressor combining top 3 models for $52,569 RMSE

## 🐛 Major Debugging Victories

### The random_state=42 Saga
**Problem**: Inconsistent train/test splits causing performance confusion across experiments  
**Solution**: Environment restart with single split using `random_state=42`  
**Learning**: Data integrity trumps model complexity - always fix random seeds first

### The OrgSize Double-Encoding Bug
**Problem**: Organization size was pre-encoded to integers, then custom transformer was binning encoded values  
**Impact**: Artificially inflated feature importance (27.8%) and poor model performance  
**Solution**: Pass raw categorical data to transformers, encode only once in pipeline

### Pipeline Integration Issues
**Problem**: Feature engineering applied outside pipeline, models trained on original dataset  
**Solution**: Custom transformer inheriting from `BaseEstimator` and `TransformerMixin`  
**Result**: All 6 engineered features properly integrated into model training

## 📊 Feature Importance Insights

1. **Organization Size (27.8%)**: Company size strongly correlates with salary ($160K → $75K gradient)
2. **Years Professional Experience (22.7%)**: Experience premium confirmed across all models
3. **Programming Languages (17.1%)**: Technology stack significantly impacts compensation
4. **Web Frameworks (6.8%)**: Specialization in modern frameworks adds salary premium
5. **Developer Type (6.5%)**: Role seniority affects compensation structure

## 🛠️ Technologies Used

- **Core**: Python, pandas, scikit-learn, NumPy
- **Models**: Random Forest, Gradient Boosting, XGBoost, Linear Regression, Ridge, ElasticNet, SVR
- **Preprocessing**: Custom transformers, ColumnTransformer, StandardScaler
- **Evaluation**: GridSearchCV, cross-validation, ensemble methods

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <github.com/hasn77/Stack-Overflow-Survey-Dataset-Model>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file listing libraries like pandas, scikit-learn, numpy, matplotlib, etc.)*

## 📁 Project Structure

```
├── data/
│   └── stack_overflow_2023_survey.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── journal.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── models.py
└── README.md
```

## 🎓 Key Learnings

- **Pipeline Design**: Proper data flow prevents encoding errors and ensures reproducibility
- **Feature Engineering**: Domain knowledge drives effective categorical variable handling
- **Debugging Mindset**: Systematic problem-solving more valuable than complex algorithms
- **Ensemble Methods**: Combining complementary algorithms captures different data patterns

## 🔄 Next Steps

- Cross-validation strategies for robust evaluation
- Feature selection techniques for optimal feature subset
- Advanced ensemble methods (stacking, blending)
- Deep learning approaches for complex pattern recognition

---

*"The answer to life, the universe, and everything is 42... and sometimes it's also the random_state that saves your ML project!"*
