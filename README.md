[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgabbygab1233%2FCrop-Recommendation&count_bg=%23161716&title_bg=%23095202&icon=leaflet.svg&icon_color=%23E7E7E7&title=Crop+Recommender&edge_flat=true)](https://hits.seeyoufarm.com)

# Crop Recommendation Using Machine Learning

Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.


<p align="center">
<img src="https://www.opendei.eu/wp-content/uploads/2020/11/img-Yanewn0ORWCx4Jlm-w800.jpg" />
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

### [Experiment Results:]()
* **Data Analysis**
    * All columns contain outliers except for N.
 * **Performance Evaluation**
    * Splitting the dataset by 80 % for training set and 20 % validation set.
 * **Training and Validation**
    * GausianNB gets a higher accuracy score than other classification models.
    * GaussianNB ( 99 % accuracy score )
 * **Performance Results**
    * Training Score: 99.5%
    * Validation Score: 99.3%

 
# Demo
Live Demo: https://ai-crop-recommender.herokuapp.com/

![](https://i.imgur.com/TnsSPQy.png)

# References
* https://www.prolim.com/crop-recommendation-system-using-machine-learning-for-digital-farming/
* https://www.irjet.net/archives/V4/i12/IRJET-V4I12179.pdf
* http://sersc.org/journals/index.php/IJAST/article/view/30399
* https://www.researchgate.net/publication/345247916_Crop_Plantation_Recommendation_using_Feature_Extraction_and_Machine_Learning_Techniques
* https://ieeexplore.ieee.org/document/8768790
* https://towardsdatascience.com/farmeasy-crop-recommendation-portal-for-farmers-48a8809b421c
* http://agri.ckcest.cn/ass/8185d605-6c4d-4d8a-b280-c867c2304d42.pdf
* https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
