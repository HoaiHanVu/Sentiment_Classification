#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as st
# Xây dựng các hàm kiểm tra kiểu dữ kiệu và hiển thị dữ liệu các cột

# Hàm lọc các cột có kiểu dữ liệu continuous
def numbers_variable(frame):
    numbers = [col for col in frame.columns if frame.dtypes[col] != object]
    return numbers

# Hàm hiển thị dữ liệu unique của các cột continuous
def display_numbers(frame, lst_numbers):
    for index, num in enumerate(lst_numbers):
        print('{}. Name var: {}, Number of unique: {}, Unique Value: {}'
          .format(index + 1, num, len(frame[num].unique()), frame[num].unique()[:10]))
        print()
        
# Hàm lọc các cột có kiểu dữ liệu categorical        
def objects_variable(frame):
    categorical = [col for col in frame.columns if frame.dtypes[col] == object]
    return categorical

# Hàm hiển thị dữ liệu unique của các cột categorical
def display_objects(frame, lst_objects):
    for index, cat in enumerate(lst_objects):
        print('{}. Name obj: {}, Number of unique: {}, Unique Values: {}'
              .format(index + 1, cat, len(frame[cat].unique()), frame[cat].unique()[:10]))
        print()

# Xây dựng các hàm phân tích dữ liệu đơn biến, hai biến đối với các thuộc tính continuous & categorcal

# Hàm phân tích đơn biến thuộc tính continuos
def continuous_analysis(frame, var):
    print('----- {} -----'.format(var))
    print(frame[var].describe())
    Q1 = np.quantile(frame[var].dropna(), 0.25)
    Q3 = np.quantile(frame[var].dropna(), 0.75)
    IQR = Q3 - Q1
    outliers = frame.loc[(frame[var] < Q1 - 1.5*IQR) | (frame[var] > Q3 + 1.5*IQR)]
    percent_outliers = outliers.shape[0] / frame[var].shape[0]
    skew = frame[var].dropna().skew()
    kurtosis = frame[var].dropna().kurtosis()
    median = frame[var].dropna().median()
    miss_value = frame[var].isnull().sum()
    print('* Median: {}'.format(median))
    print('* Skewness: {}'.format(skew))
    print('* Kurtosis: {}'.format(kurtosis))
    print('* Percentage of outliers: {}'.format(percent_outliers))
    print('* Number of missing value: {}'.format(miss_value))
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    sb.distplot(frame[var].dropna())
    plt.subplot(1, 2, 2)
    plt.boxplot(frame[var].dropna())
    plt.show()
    print()
    
# Hàm phân tích đơn biến thuộc tính categorical  
def categorical_analysis(frame, var):
    print('----- {} -----'.format(var))
    print('Describe: ')
    print(frame[var].describe())
    miss_value = frame[var].isnull().sum()
    print('* Unique value: ')
    print(frame[var].value_counts())
    print('* Mode value: {}'.format(frame[var].mode()[0]))
    print('* Number of missing value: {}'.format(miss_value))
    sb.countplot(data = frame, x = var)
    if len(frame[var].unique()) > 8:
        plt.xticks(rotation = 90)
    plt.show()
    
# Hàm phân tích hai biến có thuộc tính continuos    
def cont_cont(frame, var1, var2):
    print('----- {} vs {} -----'.format(var1, var2))
    correlation = frame[var1].corr(frame[var2])
    print('Pearson correlation between {} & {}: {}'.format(var1, var2, correlation))
    sb.pairplot(frame[[var1, var2]].dropna())
    plt.show()
    
# Hàm phân tích hai biến có thuộc tính categorical        
def cat_cat(frame, var1, var2, prob):
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    print('----- {}  vs {} -----'.format(var1, var2))
    table = pd.crosstab(frame[var1], frame[var2])
    print(table)
    plt.figure(figsize=(8, 6))
    table.plot(kind = 'bar', stacked = True)
    plt.show()
    stat, p_value, dof, expected = chi2_contingency(table)
    print('P-value: {}'.format(p_value))
    alpha = 1 - prob
    if p_value <= alpha:
        print('Reject H0 --> {} and {} are dependent.'.format(var1, var2))
    else:
        print('Accept H0 --> {} and {} are independent.'.format(var1, var2))

# Hàm phân tích ảnh hưởng của biến phân loại lên biến output(continuous)
def catvar_affected_output(frame, col, output_var):
    print('----- {} vs {} -----'.format(col, output_var))
    df = frame[[col, output_var]]
    sb.boxplot(data = df, x = col, y = output_var)
    if len(frame[col].unique()) > 8:
        plt.xticks(rotation = 90)
    df_pivot = df.pivot(columns = col, values = output_var)
    lst = []
    for column in df_pivot.columns:
        lst.append(remove_outliers(df_pivot, column))
    fvalue, pvalue = cal_anova(*lst)
    w_levene, p_levene = cal_levene(*lst)
    print('* --- Levene hypothesis --- *')
    print('p_value: {}'.format(p_levene))
    if p_levene > 0.05:
        print('Accept H0 --> Các quần thể có phương sai bằng nhau.')
    else:
        print('Reject H0 --> Các quần thể có phương sai không bằng nhau.')
    print()
    print('* --- Anova one-way hypothesis --- *')
    print('p_value: {}'.format(pvalue))
    if pvalue <= 0.05:       
        print('Reject H0 --> Có sự khác biệt đáng kể.')
        print()
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        m_comp = pairwise_tukeyhsd(endog = df[output_var],
                          groups = df[col],
                          alpha = 0.05)
        print(m_comp)
    else:
        print('Accept H0 --> Không có sự khác biệt đáng kể')

# Hàm xoá bỏ giá trị outlier bằng IQR
def remove_outliers(frame, col):
    Q1 = np.quantile(frame[col].dropna(), 0.25)
    Q3 = np.quantile(frame[col].dropna(), 0.75)
    IQR = Q3 - Q1
    clean_data = frame.loc[(frame[col] >= Q1 - 1.5*IQR) & (frame[col] <= Q3 + 1.5*IQR), col]
    return clean_data
    
# Hàm gọi tính toán Anova
def cal_anova(*arg):
    f, p = st.f_oneway(*arg)
    return f, p

# Hàm levene kiểm tra giả định Anova
def cal_levene(*arg):
    w, p_levene = st.levene(*arg)
    return w, p_levene

#Hàm so sánh hiệu suất giữa simple linear & polynomial regression model
def compare_simple_regression(frame, independ, depend, random_st, degree_poly):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    correlative = round(frame[independ].corr(frame[depend]), 4)
    
    X = frame[[independ]]
    y = frame[depend]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_st)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    m = lm.coef_[0]
    b = lm.intercept_
    regline = [m * float(x) + b for x in np.array(X)]
    yhat_test = lm.predict(X_test)
    yhat_train = lm.predict(X_train)
        
    pl = PolynomialFeatures(degree = degree_poly)
    X_pl = pl.fit_transform(X)
    X_pl_train, X_pl_test, y_pl_train, y_pl_test = train_test_split(X_pl, y, 
                                                                    random_state  = random_st)
    poly = LinearRegression()
    poly.fit(X_pl_train, y_pl_train)
    y_pl_hat_test = poly.predict(X_pl_test)
    y_pl_hat_train = poly.predict(X_pl_train)
    print('* Pearson Correlation between {} and {}: {}'.format(independ, depend, correlative))
    print()      
    print('----- LINEAR MODEL PERFORMANCE -----')
    print('R-squared model of Full: {}'.format(round(lm.score(X, y), 4)))
    print('R-squared model of Train: {}'.format(round(lm.score(X_train, y_train), 4)))
    print('R-squared model of Test: {}'.format(round(lm.score(X_test, y_test), 4)))
    print('MSE Linear of price and predicted in Train: {}'.format(mean_squared_error(y_train, yhat_train)))
    print('MAE Linear of price and predicted in Train: {}'.format(mean_absolute_error(y_train, yhat_train)))
    print('MSE Linear of price and predicted in Test: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('MAE Linear of price and predicted in Test: {}'.format(mean_absolute_error(y_test, yhat_test)))
    print()
    print('----- POLYNOMIAL MODEL PERFORMANCE -----')
    print('R-squared model of Full: {}'.format(round(poly.score(X_pl, y), 4)))
    print('R-squared model of Train: {}'.format(round(poly.score(X_pl_train, y_pl_train), 4)))
    print('R-squared model of Test: {}'.format(round(poly.score(X_pl_test, y_pl_test), 4)))
    print('MSE Polynomial of price and predicted in Train: {}'.format(mean_squared_error(y_pl_train, y_pl_hat_train)))
    print('MAE Polynomial of price and predicted in Train: {}'.format(mean_absolute_error(y_pl_train, y_pl_hat_train)))
    print('MSE Polynomial of price and predicted in Test: {}'.format(mean_squared_error(y_pl_test, y_pl_hat_test)))
    print('MAE Polynomial of price and predicted in Test: {}'.format(mean_absolute_error(y_pl_test, y_pl_hat_test)))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(X, regline, color = 'r', linewidth = 2)
    plt.scatter(X_train, y_train, color = 'green', label = 'Actual Train Values')
    plt.scatter(X_train, yhat_train, color = 'orange', label = 'Predict Train Values')
    plt.scatter(X_test, y_test, color = 'black', label = 'Actual Test Values')
    plt.scatter(X_test, yhat_test, color = 'blue', label = 'Predict Test Values')
    plt.xlabel(independ)
    plt.ylabel(depend)
    plt.legend(title = 'Notes')
    plt.subplot(1, 2, 2)
    plt.scatter(X, y)
    sb.regplot(X, poly.predict(pl.fit_transform(X)), color = 'r', fit_reg = False)
    plt.show()
    return lm, poly, X_pl

# Hàm xây dựng mô hình hồi quy tuyến tính đa biến hoàn chỉnh
def complete_multiple_linear_regression(frame, lst_independ, depend, random_st):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    X = frame[lst_independ]
    y = frame[depend]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_st)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    yhat_test = lm.predict(X_test)
    yhat_train = lm.predict(X_train)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Full: {}'.format(round(lm.score(X, y), 4)))
    print('* R-squared model of Train: {}'.format(round(lm.score(X_train, y_train), 4)))
    print('* R-squared model of Test: {}'.format(round(lm.score(X_test, y_test), 4)))
    print('* MSE Linear of output and predicted in Train: {}'.format(mean_squared_error(y_train, yhat_train)))
    print('* MAE Linear of output and predicted in Train: {}'.format(mean_absolute_error(y_train, yhat_train)))
    print('* MSE Linear of output and predicted in Test: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Linear of output and predicted in Test: {}'.format(mean_absolute_error(y_test, yhat_test)))
    print("* Pearson's correlation coefficient and p-value: {}".format(pearsonr(lm.predict(X_test), y_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_train, label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X_train), label = 'Predicted Train Values', color = 'b', ax = ax1)
    plt.legend()

    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y_test, label = 'Actual Test Values', color = 'r')
    sb.kdeplot(lm.predict(X_test), label = 'Predicted Test Values', color = 'b', ax = ax2)
    plt.legend()
    plt.show()
    return lm, X_train, X_test, y_train, y_test

# Hàm xây dựng mô hình hồi quy tuyến tính đa biến trên tập train
def multiple_linear_regression(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    lm = LinearRegression()
    lm.fit(X, y)
    yhat = lm.predict(X)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Train: {}'.format(round(lm.score(X, y), 4)))
    print('* MSE Linear of output and predicted: {}'.format(mean_squared_error(y, yhat)))
    print('* MAE Linear of output and predicted: {}'.format(mean_absolute_error(y, yhat)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y, label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X), label = 'Predicted Train Values', color = 'b', ax = ax1)
    plt.legend()
    return lm

# Hàm đánh giá model trên tập test
def eval_linear_testset(model, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    yhat_test = model.predict(X_test)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Test: {}'.format(round(model.score(X_test, y_test), 4)))
    print('* MSE Linear of output and predicted: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Linear of output and predicted: {}'.format(mean_absolute_error(y_test, yhat_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_test, label = 'Actual Test Values', color = 'r')
    sb.kdeplot(model.predict(X_test), label = 'Predicted Test Values', color = 'b', ax = ax1)
    plt.legend() 

#Hàm xây dựng mô hình hồi quy đa thức đa biến hoàn chỉnh
def complete_multiple_polynomial_regression(frame, lst_independ, depend, random_st, degree_pl):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    X = frame[lst_independ]
    y = frame[depend]
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, random_state = random_st)
    poly = LinearRegression()
    poly.fit(X1_train, y1_train)
    yhat_test = poly.predict(X1_test)
    yhat_train = poly.predict(X1_train)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Full: {}'.format(round(poly.score(X1, y), 4)))
    print('* R-squared model of Train: {}'.format(round(poly.score(X1_train, y1_train), 4)))
    print('* R-squared model of Test: {}'.format(round(poly.score(X1_test, y1_test), 4)))
    print('* MSE Polynomial of output and predicted in Train: {}'.format(mean_squared_error(y1_train, yhat_train)))
    print('* MAE Polynomial of output and predicted in Train: {}'.format(mean_absolute_error(y1_train, yhat_train)))
    print('* MSE Polynomial of output and predicted in Test: {}'.format(mean_squared_error(y1_test, yhat_test)))
    print('* MAE Polynomial of output and predicted in Test: {}'.format(mean_absolute_error(y1_test, yhat_test)))
    print("* Pearson's correlation coefficient and p-value: {}".format(pearsonr(poly.predict(X1_test), y1_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y1_train, label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1_train), label = 'Predicted Train Values', color = 'purple', ax = ax1)
    plt.legend()

    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y1_test, label = 'Actual Test Values', color = 'green')
    sb.kdeplot(poly.predict(X1_test), label = 'Predicted Test Values', color = 'purple', ax = ax2)
    plt.legend()
    plt.show()
    return poly, X1, X1_train, X1_test, y1_train, y1_test

# Hàm xây dựng mô hình hồi quy đa thức đa biến trên tập train
def multiple_polynomial_regression(X, y, degree_pl):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X)
    poly = LinearRegression()
    poly.fit(X1, y)
    yhat = poly.predict(X1)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Train: {}'.format(round(poly.score(X1, y), 4)))
    print('* MSE Polynomial of output and predicted: {}'.format(mean_squared_error(y, yhat)))
    print('* MAE Polynomial of output and predicted: {}'.format(mean_absolute_error(y, yhat)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y, label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1), label = 'Predicted Train Values', color = 'purple', ax = ax1)
    plt.legend()
    return poly, X1

# Hàm đánh giá model poly trên tập test
def eval_poly_testset(model, X_test, y_test, degree_pl):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X_test)
    yhat_test = model.predict(X1)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Test: {}'.format(round(model.score(X1, y_test), 4)))
    print('* MSE Polynomial of output and predicted: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Polynomial of output and predicted: {}'.format(mean_absolute_error(y_test, yhat_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_test, label = 'Actual Test Values', color = 'green')
    sb.kdeplot(model.predict(X1), label = 'Predicted Test Values', color = 'purple', ax = ax1)
    plt.legend()
    
# Hàm xây dựng mô hình hồi quy tuyến tính với 2 output hoàn chỉnh
def complete_multiple_2output_linear_regression(frame, lst_independ, depend, random_st):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    X = frame[lst_independ]
    y = frame[depend]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_st)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    yhat_test = lm.predict(X_test)
    yhat_train = lm.predict(X_train)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Full: {}'.format(round(lm.score(X, y), 4)))
    print('* R-squared model of Train: {}'.format(round(lm.score(X_train, y_train), 4)))
    print('* R-squared model of Test: {}'.format(round(lm.score(X_test, y_test), 4)))
    print('* MSE Linear of output and predicted in Train: {}'.format(mean_squared_error(y_train, yhat_train)))
    print('* MAE Linear of output and predicted in Train: {}'.format(mean_absolute_error(y_train, yhat_train)))
    print('* MSE Linear of output and predicted in Test: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Linear of output and predicted in Test: {}'.format(mean_absolute_error(y_test, yhat_test)))
    #print("* Pearson's correlation coefficient and p-value: {}".format(pearsonr(lm.predict(X_test), y_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_train.iloc[:, 0], label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X_train)[:, 0], label = 'Predicted Train Values', color = 'b', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y_test.iloc[:, 0], label = 'Actual Test Values', color = 'r')
    sb.kdeplot(lm.predict(X_test)[:, 0], label = 'Predicted Test Values', color = 'b', ax = ax2)
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_train.iloc[:, 1], label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X_train)[:, 1], label = 'Predicted Train Values', color = 'b', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y_test.iloc[:, 1], label = 'Actual Test Values', color = 'r')
    sb.kdeplot(lm.predict(X_test)[:, 1], label = 'Predicted Test Values', color = 'b', ax = ax2)
    plt.legend()
    plt.show()
    return lm, X_train, X_test, y_train, y_test

# Hàm xây dựng mô hình hồi quy tuyến tính với 2 output trên tập train
def multiple_2output_linear_regression(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    lm = LinearRegression()
    lm.fit(X, y)
    yhat = lm.predict(X)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Train: {}'.format(round(lm.score(X, y), 4)))
    print('* MSE Linear of output and predicted in Train: {}'.format(mean_squared_error(y, yhat)))
    print('* MAE Linear of output and predicted in Train: {}'.format(mean_absolute_error(y, yhat)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y.iloc[:, 0], label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X)[:, 0], label = 'Predicted Train Values', color = 'b', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y.iloc[:, 1], label = 'Actual Train Values', color = 'r')
    sb.kdeplot(lm.predict(X)[:, 1], label = 'Predicted Train Values', color = 'b', ax = ax2)
    plt.legend()
    plt.show()
    return lm

# Hàm đánh giá linear model 2 output trên tập test
def eval_2output_linear_testset(model, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    yhat_test = model.predict(X_test)
    print('----- MULTIPLE LINEAR REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Test: {}'.format(round(model.score(X_test, y_test), 4)))
    print('* MSE Linear of output and predicted in Test: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Linear of output and predicted in Test: {}'.format(mean_absolute_error(y_test, yhat_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_test.iloc[:, 0], label = 'Actual Test Values', color = 'r')
    sb.kdeplot(model.predict(X_test)[:, 0], label = 'Predicted Test Values', color = 'b', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y_test.iloc[:, 1], label = 'Actual Test Values', color = 'r')
    sb.kdeplot(model.predict(X_test)[:, 1], label = 'Predicted Test Values', color = 'b', ax = ax2)
    plt.legend()
    plt.show()


# Hàm xây dựng mô hình hồi quy đa thức đa biến với 2 output hoàn chỉnh
def complete_multiple_2output_polynomial_regression(frame, lst_independ, depend, random_st, degree_pl):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats.stats import pearsonr
    X = frame[lst_independ]
    y = frame[depend]
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, random_state = random_st)
    poly = LinearRegression()
    poly.fit(X1_train, y1_train)
    yhat_test = poly.predict(X1_test)
    yhat_train = poly.predict(X1_train)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Full: {}'.format(round(poly.score(X1, y), 4)))
    print('* R-squared model of Train: {}'.format(round(poly.score(X1_train, y1_train), 4)))
    print('* R-squared model of Test: {}'.format(round(poly.score(X1_test, y1_test), 4)))
    print('* MSE Polynomial of output and predicted in Train: {}'.format(mean_squared_error(y1_train, yhat_train)))
    print('* MAE Polynomial of output and predicted in Train: {}'.format(mean_absolute_error(y1_train, yhat_train)))
    print('* MSE Polynomial of output and predicted in Test: {}'.format(mean_squared_error(y1_test, yhat_test)))
    print('* MAE Polynomial of output and predicted in Test: {}'.format(mean_absolute_error(y1_test, yhat_test)))
    #print("* Pearson's correlation coefficient and p-value: {}".format(pearsonr(poly.predict(X1_test), y1_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y1_train.iloc[:, 0], label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1_train)[:, 0], label = 'Predicted Train Values', color = 'purple', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y1_test.iloc[:, 0], label = 'Actual Test Values', color = 'green')
    sb.kdeplot(poly.predict(X1_test)[:, 0], label = 'Predicted Test Values', color = 'purple', ax = ax2)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y1_train.iloc[:, 1], label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1_train)[:, 1], label = 'Predicted Train Values', color = 'purple', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y1_test.iloc[:, 1], label = 'Actual Test Values', color = 'green')
    sb.kdeplot(poly.predict(X1_test)[:, 1], label = 'Predicted Test Values', color = 'purple', ax = ax2)
    plt.legend()
    plt.show()
    return poly, X1, X1_train, X1_test, y1_train, y1_test

#  Hàm xây dựng mô hình hồi quy đa thức với 2 output trên tập train
def multiple_2output_polynomial_regression(X, y, degree_pl):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X)
    poly = LinearRegression()
    poly.fit(X1, y)
    yhat = poly.predict(X1)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Train: {}'.format(round(poly.score(X1, y), 4)))
    print('* MSE Polynomial of output and predicted in Train: {}'.format(mean_squared_error(y, yhat)))
    print('* MAE Polynomial of output and predicted in Train: {}'.format(mean_absolute_error(y, yhat)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y.iloc[:, 0], label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1)[:, 0], label = 'Predicted Train Values', color = 'purple', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y.iloc[:, 1], label = 'Actual Train Values', color = 'green')
    sb.kdeplot(poly.predict(X1)[:, 1], label = 'Predicted Train Values', color = 'purple', ax = ax2)
    plt.legend()
    plt.show()
    return poly, X1

# Hàm đánh giá poly model 2 output trên tập test
def eval_2output_poly_testset(model, X_test, y_test, degree_pl):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    pl = PolynomialFeatures(degree = degree_pl)
    X1 = pl.fit_transform(X_test)
    yhat_test = model.predict(X1)
    print('----- MULTIPLE POLYNOMIAL REGRESSION MODEL PERFORMANCE -----')
    print('* R-squared model of Test: {}'.format(round(model.score(X1, y_test), 4)))
    print('* MSE Polynomial of output and predicted in Test: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE Polynomial of output and predicted in Test: {}'.format(mean_absolute_error(y_test, yhat_test)))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax1 = sb.kdeplot(y_test.iloc[:, 0], label = 'Actual Test Values', color = 'green')
    sb.kdeplot(model.predict(X1)[:, 0], label = 'Predicted Test Values', color = 'purple', ax = ax1)
    plt.legend()
    plt.subplot(1, 2, 2)
    ax2 = sb.kdeplot(y_test.iloc[:, 1], label = 'Actual Test Values', color = 'green')
    sb.kdeplot(model.predict(X1)[:, 1], label = 'Predicted Test Values', color = 'purple', ax = ax2)
    plt.legend()
    plt.show()

def crossval_linear_regression(frame, lst_independ, depend, cv):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    X = frame[lst_independ]
    y = frame[depend]
    model = LinearRegression()
    Rcross = cross_val_score(model, X, y, cv = cv)
    MSEcross = cross_val_score(model, X, y, cv = cv, scoring = 'neg_mean_squared_error') * (-1)
    print('----- CROSS VALIDATION OF MULTIPLE LINEAR REGRESSION PREFOMANCE -----')
    print('* Rcross value: {}'.format(Rcross.tolist()))
    print('* Mean of folds: {}'.format(round(Rcross.mean(), 4)))
    print('* Std of folds: {}'.format(round(Rcross.std(), 4)))
    print()
    print('* MSE value: {}'.format(MSEcross.tolist()))
    print("* Mean's MSE of fold: {}".format(round(MSEcross.mean(), 4)))
    print("* Std's MSE of fold: {}".format(round(MSEcross.std(), 4)))
    return Rcross

# Hàm xây dựng mô hình dự đoán hoàn chỉnh
def complete_logistic_regression(frame, lst_predictor, predict, random_st, testsize):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    X = frame[lst_predictor]
    y = frame[predict]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_st, test_size = testsize)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)
    print('----- LOGISTIC REGRESSION MODEL PREFOMANCE -----')
    print('* R-squared model of Full: {}'.format(round(model.score(X, y), 4)))
    print('* R-squared model of Train: {}'.format(round(accuracy_score(y_train, yhat_train), 4)))
    print('* R-squared model of Test: {}'.format(round(accuracy_score(y_test, yhat_test), 4)))
    print('* Confusion Matrix of Train: ')
    print(confusion_matrix(y_train, yhat_train))
    print('* Classification Report of Train: ')
    print(classification_report(y_train, yhat_train))
    print('* Confusion Matrix of Test: ')
    print(confusion_matrix(y_test, yhat_test))
    print('* Classification Report of Test: ')
    print(classification_report(y_test, yhat_test))
    return model

# Hàm xây dựng mô hình dự đoán trên tập train
def logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    model = LogisticRegression()
    model.fit(X, y)
    yhat_train = model.predict(X)
    print('----- LOGISTIC REGRESSION MODEL PREFOMANCE -----')
    print('* R-squared model of Train: {}'.format(round(accuracy_score(y, yhat_train), 4)))
    print()
    print('* Confusion Matrix of Train: ')
    print(confusion_matrix(y, yhat_train))
    print()
    print('* Classification Report of Train: ')
    print(classification_report(y, yhat_train))
    return model

# Hàm đánh giá mô hình trên tập test
def eval_logit_testset(model, X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    yhat_test = model.predict(X)
    print('----- LOGISTIC REGRESSION MODEL PREFOMANCE -----')
    print('* R-squared model of Test: {}'.format(round(accuracy_score(y, yhat_test), 4)))
    print()
    print('* Confusion Matrix of Test: ')
    print(confusion_matrix(y, yhat_test))
    print()
    print('* Classification Report of Test: ')
    print(classification_report(y, yhat_test))

# Hàm vẽ đường cong ROC theo biến phân loại tuỳ chọn
def ROC_curve_display(model, X, y, pred):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    yhat_proba = model.predict_proba(X)
    print('* Area below the curve: {}'.format(round(roc_auc_score(y, yhat_proba[:, pred]), 5)))
    print()
    fpr, tpr, thresholds = roc_curve(y, yhat_proba[:, pred])
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, marker = '.')
    plt.xlabel('False Possitve Rate')
    plt.ylabel('True Possitive Rate')
    plt.title('ROC Curve of Predict {}'.format(pred))
    plt.show()


# In[ ]:




