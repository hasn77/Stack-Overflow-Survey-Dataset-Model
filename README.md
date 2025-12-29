# Stack Overflow Developer Salary Prediction

A complete end-to-end machine learning project predicting developer salaries using the Stack Overflow 2023 Survey dataset.

**Goal**: Predict yearly developer salaries (ConvertedCompYearly) using survey responses  
**Dataset**: Stack Overflow 2023 Survey (~48K salary records)  
**Final Performance**: **$52,569 RMSE** using ensemble methods

## Key Results

- **88% Performance Improvement**: From $432K RMSE (initial disaster) to $52K RMSE (final ensemble)
- **Advanced Pipeline**: Custom transformers, target encoding, and leak-proof preprocessing
- **Ensemble Victory**: VotingRegressor combining Random Forest, and Gradient Boosting

## Technical Architecture

### Data Preprocessing Pipeline
- **Multi-label Categorical Handling**: Processed semicolon-separated survey responses (languages, platforms, tools)
- **Strategic Bucketization**: Reduced high-cardinality features using domain knowledge
- **Custom Transformers**: Built `AdvancedFeatureEngineer` and `OrgSizeBinner` for pipeline integration
- **Core**: Python, pandas, scikit-learn, NumPy
- **Models**: Random Forest, Gradient Boosting, XGBoost, Linear Regression, Ridge, ElasticNet, SVR
- **Preprocessing**: Custom transformers, ColumnTransformer, StandardScaler
- **Evaluation**: GridSearchCV, cross-validation, ensemble methods

## Key Learnings

- **Pipeline Design**: Proper data flow prevents encoding errors and ensures reproducibility
- **Feature Engineering**: Domain knowledge drives effective categorical variable handling
- **Debugging Mindset**: Systematic problem-solving more valuable than complex algorithms
- **Ensemble Methods**: Combining complementary algorithms captures different data patterns

## Next Steps

- Cross-validation strategies for robust evaluation
- Feature selection techniques for optimal feature subset
- Advanced ensemble methods (stacking, blending)
- Deep learning approaches for complex pattern recognition
