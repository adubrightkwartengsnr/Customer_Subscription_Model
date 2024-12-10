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

| ![data distribution](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20122847.png)         | ![outlier distribution](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20122925.png) |

-------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |

|  ![campaigns by subscription](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20123157.png) | ![subscriptions by month](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20123308.png) | 

-------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![campaign trend vs subcription rate](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20123324.png) | ![target class distribution](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20124651.png) |

## Model Selection


| ![Model](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20130211.png) |
![Feature-importance](https://github.com/adubrightkwartengsnr/Customer_Subscription_Model/blob/dev/eda%20assets/Screenshot%202024-12-09%20212059.png)

After carefully assessing the performance of our models using key evaluation metrics, it is evident that the XGBClassifier and Gradient Boosting models stand out as the most effective choices for our dataset. The F1-score serves as a crucial indicator, and the models achieved the highest F1-score of 0.9 among all models evaluated. This indicates that the XGBClassifier and the Gradient Boosting models provide the most accurate and precise predictions when compared to RandomForest, and Logistic Regression models.

Therefore, for this specific prediction task, we are adopting the XGBClassifier and Gradient Boosting models for its superior predictive accuracy.


## Recommendations

1. **Campaign Optimization:** Based on the analysis of the impact of number of campaigns on subscription, the marketing team should tailor promotional campaigns accordingly. I think they should limit the number of campaigns per person.

2. **Focus on customers that previously subscribed to a term**: The customers with an outcome, "Success," in the previous campaings stands out with the highest chance of subscribing to the latest promotions. It's essential to allocate additional resources and marketing efforts to maintain and potentially improve success rate for these customers. Additionally, cellular contacts made to customers yielded a lot of positive outcomes.We should Consider improving customer-contact strategies to capitalize on these markets.

3. **Incentive Programs**: The bank should offer incentives for first-time subscribers or loyalty programs for returning customers, such as discounts on deposit rates, rewards points, or exclusive offers.

4. **Data-driven timing optimizayion**: Februray to March,August-September and November-December yield better responses based on data (e.g., targeting clients in high-conversion months).The marketing team should conduct campaigns during these optimal periods to maximize engagement and minimize wasted effort.

## Getting Started🏁


<!-- GETTING STARTED -->

## 💻 Getting Started <a name="getting-started"></a>


To get a local copy up and running, follow these steps.

### Prerequisites

In order to run this project you need:

- Python


### Setup

Clone this repository to your desired folder:


```sh
  cd my-folder
  git clone https://github.com/adubrightkwartengsnr/Customer_Subscription_Model
```

Change into the cloned repository

```sh
  cd Customer_Subscription_Model
  
```

Create a virtual environment

```sh

python -m venv env

```

Activate the virtual environment

```sh
    env/Scripts/activate
```


### Install

Here, you need to recursively install the packages in the `requirements.txt` file using the command below 

```sh
   pip install -r requirements.txt
```

### Prerequisites

In order to run this project, you need:

- Python installed on your machine

## License📜

This project is licensed under the [MIT License](LICENSE).

## Author✍️

Bright Kwarteng Senior Adu

Connect with me on LinkedIn: [LinkedIn Profile](https://linkedin.com/in/bright-adu-kwarteng-snr)

---

Feel free to star ⭐ this repository if you find it helpful!

