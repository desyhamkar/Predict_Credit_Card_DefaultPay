# Final Project of Data Mining Course - Big Data Analysis
# Predict_Credit_Card_DefaultPay
---
 - categories: 
 - machine learning
 - date: "2020-01-24"
 - title: Predict Credit Card Pay - Final Project of Data Mining Course - Big Data Analysis
 ---



default of credit card clients Data Set
Sources
URL : https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
This research aimed at the case of customer default payments in Taiwan and compares the predictive accuracy of probability , sensitivity, Specificity ,lift, AUC ( Area Under the curve) of default among these data mining methods. There are :
1. Regression(LinearRegression,LogisticRegression)
2. kNN ( K-Nearest Neighbors)
3. SVM ( Support Vector Machine)
4. DecisionTree
5. ModelComparison
Our Data contains : 3000 observation and 24 features/variables 

 
# Attribute Information

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:		
Attribute	Description	Value


* **X1**	 Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.	 Amount of the given credit (dollar)
* **X2**	Gender 	(1 = male; 2 = female).
* **X3**	Education 	(1 = graduate school; 2 = university; 3 = high school; 4 = others).
* **X4**	Marital status 	(1 = married; 2 = single; 3 = others).
* **X5**	Age 	(year)
* **X6**	History of past payment. Tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; 	The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
* **X7**	the repayment status in Agustus  2005	
* **X8**	the repayment status in July, 2005	
* **X9**	the repayment status in June, 2005	
* **X10**	the repayment status in Mei, 2005	
* **X11**	the repayment status in April, 2005. 	
* **X12**	amount of bill statement in September, 2005.	Amount of bill statement (NT dollar). 
* **X13**	amount of bill statement in August, 2005.	
* **X14**	amount of bill statement in July, 2005.	
* **X15**	amount of bill statement in June, 2005.	
* **X16**	amount of bill statement in Mei, 2005.	
* **X17**	amount of bill statement in April, 2005.	
* **X18**	amount paid in September, 2005.	X18-X23:Amount of previous payment (NT dollar).
* **X19**	amount paid in August, 2005.	
* **X20**	amount paid in July, 2005.	
* **X21**	amount paid in June, 2005.	
* **X22**	amount paid in Mei, 2005.	
* **X23**	amount paid in April, 2005.	
* **Y**	default payment next month	Target Variable,  default payment (Yes = 1, No = 0)
