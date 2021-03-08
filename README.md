# Crop-Recommendation

Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.


<p align="center">
<img src="https://lh3.googleusercontent.com/proxy/o-uhVvrI2YPCMYyTP4-iGBWqIreujotbLF_CchfKnqddsEoFnIhl0aQ5VuXjDUKBa6bCkN084wXqu29uUthZzCm0pY-073dtTD8xGJ4" />
</p>
This application will assist farmers in increasing agricultural productivity, preventing soil degradation in cultivated land, reducing chemical use in crop production, and maximizing water resource efficiency.

# [Dataset]()
This dataset was build by augmenting datasets of rainfall, climate and fertilizer data available for India.

### [Attributes information:]()

* **N** - Ratio of Nitrogen content in soil
* **P** - Ratio of Phosphorous content in soil
* **K** - Ratio of Potassium content in soil
* **Temperature** -  temperature in degree Celsius
* **Humidity** - relative humidity in %
* **ph** - ph value of the soil
* **Rainfall** - rainfall in mm 

### Experiment Results:
* **Data Analysis**
    * 5 columns contains outliers this columns are ( creatitinine_phosphokinase, ejection_fraction, platelets, sereum_creatinine, serum_soidum).
    * Imbalanced target class ( I'll used resampling techniques to add more copies of the minority class )
 * **Performance Evaluation**
    * Splitting the dataset by 80 % for training set and 20 % validation set.
 * **Training and Validation**
    * After training and experimenting different algorithms using ensemble models have good accuracy score than linear and nonlinear models.
    * Gradient Boosting Classifier ( 93 % accuracy score )
 * **Fine Tuning**
    * Using {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 1000, 'subsample': 0.7} for Gradient Boosting Classifier improved the accuracy by 1 %.
 * **Performance Results**
    * Validation Score: 97%
    * ROC_AUC Score: 96.9 %
 
# Demo


# References
* https://www.prolim.com/crop-recommendation-system-using-machine-learning-for-digital-farming/
* https://www.irjet.net/archives/V4/i12/IRJET-V4I12179.pdf
* http://sersc.org/journals/index.php/IJAST/article/view/30399
* https://www.researchgate.net/publication/345247916_Crop_Plantation_Recommendation_using_Feature_Extraction_and_Machine_Learning_Techniques
* https://ieeexplore.ieee.org/document/8768790
* https://towardsdatascience.com/farmeasy-crop-recommendation-portal-for-farmers-48a8809b421c
* http://agri.ckcest.cn/ass/8185d605-6c4d-4d8a-b280-c867c2304d42.pdf
* https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
