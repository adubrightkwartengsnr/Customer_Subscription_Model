# 💼 Customer Subscription Model Learning Prediction Project
The key objective of this project is to create a machine learning model capable to predict whether a given customer will subscribe to a serveice or not based on features of the bank dataset.The model is developed by training the algorithm on a dataset of historical data. The algorithm learns from the data and identifies patterns that can be used to predict the value of the dependent variable.

Once the model is trained, it can be used to predict the value of the dependent variable for new data points. This can be used to make decisions about future outcomes, such as creating targeted campaigns, improving upon future subscriptions, or helping making sales and marketting staff make more data-informed decisions.

## Project Overview

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to explore and analyze the sales data. In this project, we'll predict customer subscritpion outcome on data from **Bank Dataset**, a data obtained from customer activities in a bank.

Specifically, we are to build a model that more accurately predicts the subscritpion outcome of customers based on the input dataset.


## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Data Dictionary](#data-dictionary)
- [Project Highlights](#project-highlights)
- [Summary](#summary)
- [Hypothesis Investigated](#hypothesis-investigated)
  - [Results](#results)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Selection](#model-selection)
- [Recommendations](#recommendations)
- [Getting Started](#getting-started)
- [License](#license)
- [Author](#author)

## Project Structure📂

- `api/`: Contains the Application Programming Interface built with FASTAPI to provide an endpoint that applications can connect to communicate with the two best performing models. This repo also contains Dockerfile and requirements.txt file for containerizing the application for crossplatform deployment.
- `data/`: Contains the dataset used for analysis, model training and testing 
- `notebooks/`Contains Jupyter notebook detailing the data exploration, preprocessing, and model building steps.
- `docker-compose file/`: Holds yaml based script for building the Docker images.
- `LICENSE`: Project license.
- `README.md`: Project overview, links, highlights, and information.
`requirements.txt/`: Houses all the libraries and dependencies with specific versions used in building the entire project. 

## Data Dictionary


| **Variable Name** | **Description**                                                  | **Data Type** | **Possible Values**                                                                                              |
|--------------------|------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------|
| **age**            | Age of the client                                               | Numeric       | Any positive integer (e.g., 18, 35, etc.)                                                                        |
| **job**            | Type of job                                                     | Categorical   | "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services" |
| **marital**        | Marital status of the client                                    | Categorical   | "married", "divorced", "single"                                                                                  |
| **education**      | Level of education                                              | Categorical   | "unknown", "secondary", "primary", "tertiary"                                                                    |
| **default**        | Has credit in default?                                          | Binary        | "yes", "no"                                                                                                      |
| **balance**        | Average yearly balance of the client (in euros)                 | Numeric       | Any real number (e.g., -500, 0, 1500)                                                                            |
| **housing**        | Does the client have a housing loan?                            | Binary        | "yes", "no"                                                                                                      |
| **loan**           | Does the client have a personal loan?                           | Binary        | "yes", "no"                                                                                                      |
| **contact**        | Contact communication type used                                 | Categorical   | "unknown", "telephone", "cellular"                                                                               |
| **day**            | Last contact day of the month                                   | Numeric       | 1–31                                                                                                             |
| **month**          | Last contact month of the year                                  | Categorical   | "jan", "feb", "mar", ..., "nov", "dec"                                                                           |
| **duration**       | Duration of the last contact, in seconds                        | Numeric       | Any non-negative integer (e.g., 0, 120, 300)                                                                     |
| **campaign**       | Number of contacts performed during this campaign               | Numeric       | Any positive integer (e.g., 1, 2, 10)                                                                            |
| **pdays**          | Number of days since the client was last contacted in a previous campaign | Numeric       | -1 (not previously contacted), or any positive integer (e.g., 5, 30, etc.)                                       |
| **previous**       | Number of contacts performed before this campaign               | Numeric       | Any non-negative integer (e.g., 0, 1, 5)                                                                         |
| **poutcome**       | Outcome of the previous marketing campaign                      | Categorical   | "unknown", "other", "failure", "success"                                                                         |
| **y**              | Target variable: has the client subscribed to a term deposit?   | Binary        | "yes", "no"                                                                                                      |



# Project Highlights🚀

- Employed a holistic approach, embracing the CRISP-DM framework, to gain a deep understanding of customer dynamics.
- Mined invaluable insights from extensive exploratory data analysis, unveiling hidden trends and patterns within the dataset.
- Engineered advanced predictive models, featuring the formidable XGBoost algorithm, to forecast sales with unprecedented accuracy.
- Implemented rigorous hyperparameter tuning, unlocking the full potential of our models and achieving unparalleled predictive performance.
- Crafted a compelling and informative README file, sharing the project's compelling journey, groundbreaking results, and its potential to reshape the future of retail forecasting.

## Summary

| Code | Name                                     |                                             Published Article                                              |                                                                                                                                                    Deployed Dashboard |
| ---- | ---------------------------------------- | :--------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Talent Mobility Program(Project) | Customer Subcription Model| [GitHub Link](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model) | [Deployed API lINK](https://customer-subscription-model-3.onrender.com) |

## Hypothesis Investigated

**Null Hypothesis (H0)** :The number of campaigns sent to a client does not significantly impact their likelihood of subscribing to a term deposit.

**Alternate Hypothesis (Ha)** : The number of campaigns sent to a client significantly impacts their likelihood of subscribing to a term deposit.

## Rationale

The rationale for testing these hypotheses is to determine whether there is empirical evidence to support the idea that the number of campaigns sent to a particular customer have a meaningful impact on subscription outcome. 

By testing these hypotheses and examining the correlation between campaigns and term deposit subscription, the bank can gain valuable insights and make evidence-based decisions regarding their campaign strategies.


### Results

| Test Conducted               | scipy.stats.chi2_contingency    | P-Value                |
| ---------------------------- | ------------------ | ---------------------- |
| Pearson's chi-squared statistic | 359.492 | 9.89613260623402e-50|

In conclusion, the scipy.stats.chi2_contingency is a hypothesis test that determines if two variables are independent of each other. It uses a contingency table, which is a two-way table that shows the observed frequencies for different combinations of the two variables. The test compares the number of campaigns sent to the customers who subscribed as against those who didn't subscribed treating them as independent variables. The test compares the observed and expected frequencies of the subjects. The expected frequencies are calculated based on the multiplication rule of probability to construct the contingency table. With chi-square statistic of 359.492. The corresponding p-value obtained from the  analysis is very close to zero (P-value:9.89613260623402e-50). Based on the results of this analysis, we reject the null hypothesis.

There is a statistically significant positive relationship (359.492) between the number of campaigns and subscription term deposit outcome. This suggests that the number of campaigns have a significant influence on subscription term deposit, and as the number of campaigns increases, subscription rate decreases as well.



## Exploratory Data Analysis (EDA)📊


A snapshot of the conducted exploratory data analysis, aimed at addressing pivotal business inquiries during the analysis process.

| ![storesbytype](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/dae0298b-2477-4772-a650-31a74b839266)         | ![storesbystate](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/549cf603-45e6-43a6-9304-760c36c5f324)       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![oil trend](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/468f4fe4-84d4-4e7a-a58a-d85739646a49) | ![newplot](https://github.com/snyamson/LP3-Super-Store-Time-Series-Forecasting/assets/58486437/79b272b7-319c-48de-901c-467ca57b9a78) |

## Model Selection


![Model](https://github.com/Makafui-Kwawu/Time-Series-Regression-Analysis-Corporation_Favorita/assets/160020850/02e79d25-7683-4e5f-86c1-4ade6ea44ae4)

After carefully assessing the performance of our models using key evaluation metrics, it is evident that the XGBoost model stands out as the most effective choice for our dataset. The RMSLE (Root Mean Squared Logarithmic Error) serves as a crucial indicator, and the XGBoost model achieved the lowest RMSLE of 0.007 among all models evaluated. This indicates that the XGBoost model provides the most accurate and precise predictions when compared to ARIMA, SARIMA, and Linear Regression models.

Therefore, for this specific forecasting task, we are adopting the XGBoost model for its superior predictive accuracy.


## Recommendations

1. **Promotion Optimization:** Based on the analysis of the impact of promotions on sales, consider optimizing promotion strategies. Identify which types of promotions (e.g., discounts, BOGO offers) have the most significant influence on sales and tailor promotional campaigns accordingly. By focusing promotional efforts on what truly drives sales, you can maximize the return on investment.

2. **Focus on High-Performing Cities**: The top-performing city, "Quito," stands out with the highest sales. It's essential to allocate additional resources and marketing efforts to maintain and potentially increase sales in Quito. Additionally, cities like "Guayaquil," "Cuenca," "Ambato," and "Santo Domingo" have also shown strong sales performance. Consider developing city-specific strategies to capitalize on these markets.

3. **Cluster-Centric Approach**: The analysis reveals that certain clusters, such as "Cluster 14," "Cluster 6," and "Cluster 8," exhibit remarkable sales figures. Invest in understanding the unique characteristics of these clusters and tailor product assortments, promotions, and inventory management strategies to maximize sales potential in these areas.

4. **Cross-Analysis Opportunities**: Explore opportunities to combine the strengths of high-performing cities, clusters, store types, and states. For example, consider aligning promotions with holidays and events in top cities and clusters to maximize sales impact. Additionally, assess whether specific store types thrive in particular cities or clusters, and use this information to refine expansion plans.

## Getting Started🏁

1. Clone this repository: `[git clone https://github.com/Makafui-Kwawu/Time-Series-Regression-Analysis-Corporation_Favorita.git]`
2. Navigate to the project directory: `LP3-Super-Store-Time-Series-Forecasting`
3. Explore the Jupyter notebooks for detailed steps and code execution.
4. Read the published article for a comprehensive understanding of the project.

## License📜

This project is licensed under the [MIT License](LICENSE).

## Author✍️

Success Makafui Kwawu

Connect with me on LinkedIn: [LinkedIn Profile](https://linkedin.com/in/smkwawu/)

---

Feel free to star ⭐ this repository if you find it helpful!

