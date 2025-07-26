# ClinicalClusterAI
An Intelligent Patient Stratification System. Developed an end-to-end machine learning pipeline that Identifies clinically distinct patient groups, Predicts patient segments for new admissions, Explains model decisions.

#  ClinicalClusterAI: Intelligent Patient Stratification System

##  Overview
An end-to-end machine learning system that:
1. **Automatically segments** hospital patients into clinically distinct groups using K-Means clustering
2. **Predicts patient categories** for new admissions with 94% accuracy using SVM
3. **Explains predictions** through SHAP value visualizations

##  Key Insights
Discovered 5 patient cohorts with distinct characteristics:
| Cluster | Profile | Avg Stay | Readmission Risk |
|---------|---------|----------|------------------|
| 0 | ChronicCare_Obese | 9.2 days | 32% |
| 1 | RoutineCare_Stable | 3.1 days | 8% |
| 2 | Diagnostics | 5.7 days | 15% |
| 3 | HighRisk_Hypertension | 11.5 days | 47% |
| 4 | UrgentCare_Diabetes | 7.8 days | 28% |

##  Technical Implementation
```mermaid
graph TD
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[K-Means Clustering]
    D --> E[SVM Classification]
    E --> F[Streamlit Dashboard]
