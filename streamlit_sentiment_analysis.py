import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import import_ipynb
import Functions as fc

# 1. Read data:
df = pd.read_csv('data_Foody.csv')

st.title('SENTIMENT ANALYSIS PROJECT')
st.header("Foody's Review Classification")

# 2. Preprocessing
def convert_score(score):
    if score >= 5.0:
        return 'Good'
    else:
        return 'Bad'    
df['class'] = df.apply(lambda x: convert_score(x['review_score']), axis = 1)
df['review_score'] = np.round(df['review_score'].astype('float'), 2)

encoder = LabelEncoder()
df['labels'] = encoder.fit_transform(df['class'])

X = df.iloc[:, 2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 3. Build model

logit = LogisticRegression(random_state = 42)
model = logit.fit(X_train_cv, y_train)

# 4. Load model & evaluate:

train_eval = model.score(X_train_cv, y_train)
test_eval = model.score(X_test_cv, y_test)
report = classification_report(y_test, model.predict(X_test_cv))
conf_matrix = confusion_matrix(y_test, model.predict(X_test_cv))

y_score = cross_val_predict(model, X_test_cv, y_test, cv = 3, method = 'decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)



# 5. GUI:
menu = ['Business Objective', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write('### 1. Data Overview:')
    st.dataframe(df.iloc[:, 2:].head(5))
    st.dataframe(df.iloc[:, 2:].tail(5))
    st.write('### 2. Visualization: ')
    score = df['review_score'].value_counts()
    score = pd.DataFrame(score).sort_values(by = 'review_score', ascending = False)

    st.write('##### Phổ điểm đánh giá của khách hàng: ')
    plt.figure(figsize = (20, 6))
    fig1 = sb.barplot(score.index, score['review_score'])
    st.pyplot(fig1.figure)
    st.write("""Dễ dàng nhận thấy điểm số từ 7.0 trở lên chiếm số lượng khá lớn, dựa trên số điểm có thể phỏng đoán các đánh giá thực sự chê hoặc nhận xét không tốt về dịch vụ chiếm số lượng nhỏ hơn các đánh giá tích cực khá nhiều.""")

    st.write('##### Số lượng review của 2 nhóm')
    plt.figure(figsize = (10, 6))
    fig2 = sb.countplot(data = df[['class']], x = 'class')
    st.pyplot(fig2.figure)
    st.write("""
    1. Việc đánh giá và cho điểm của khách hàng dựa khá nhiều vào cảm xúc cũng như nhiều yếu tố khác nhau cộng hưởng đến trải nghiệm dịch vụ. Thực tế từ dữ liệu `review_text` và `score` cho thấy, việc khen chê trong cùng một post đánh giá (review) khá phổ biến, vì như giải thích ở trên, có khá nhiều yếu tố tác động đến trải nghiệm của khách hàng sử dụng dịch vụ như: giá cả, chất lượng, vị trí, phục vụ, thái độ nhân viên, khẩu vị, ... . Vì vậy, trong đa số trường hợp, việc một bình luận của khách hàng không hẳn khớp với mức điểm đánh giá có thể hiểu được bằng việc khách hàng thích điểm này nhưng không thích điểm kia của nhà hàng. Hơn nữa, với những khách hàng khác nhau sẽ có những cảm nhận khác nhau, cách cho điểm đối với từng yếu tố cụ thể giữa hai người cũng sẽ khác nhau.


    2. Từ những giả định trên, ta sẽ chọn ra một ngưỡng điểm để phân loại các đánh giá và bình luận. Ngưỡng điểm này sẽ không quá cao trên phổ điểm từ 1 - 10, dựa trên những lập luận ở trên. Với các số điểm vượt trên ngưỡng, ta có thể xem nhà hàng được đánh giá có chất lượng ở mức ổn đến tốt, có thể tồn tại cả điểm hài lòng và chưa hài lòng nhưng nhìn chung là đánh giá ổn. Ngược lại, với điểm dưới ngưỡng, có thể xem nhà hàng nhận đánh giá khá tiêu cực và cần nhìn nhận lại cách cung cấp dịch vụ nói chung.
    3. Ta sẽ chọn ngưỡng điểm là 5 để phân loại các bình luận.""")
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for positive and negative classification.""")
elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('#### 1. Model Summary')
    st.write(""" Using Logistic Regression Algorithm to build classify model.""")
    st.write('#### 2. Model Evaluation: ')
    st.write('##### Training history: ')
    st.write('##### Train Score: ')
    st.code(' - Accuracy: ' + str(np.round(train_eval, 3)))
    st.write('##### Valid Score: ')
    st.code(' - Accuracy: ' + str(np.round(test_eval, 3)))
    st.write('##### Classification Report: ')
    st.code(report)
    st.write('##### Confusion Matrix: ')
    st.code(conf_matrix)

    st.write('#### Precision and Recall for each Threshold: ')
    plt.figure(figsize=(10, 6))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fc.plot_precision_recall_curve(model, X_test_cv, y_test, cv = 3))

    st.write('#### Precision and Recall: ')
    st.pyplot(plot_precision_vs_recall(precisions, recalls))

    st.write('#### ROC curve of class `1`: ')
    plt.figure(figsize=(10, 6))
    st.pyplot(fc.ROC_curve_display(model, X_test_cv, y_test, 1))
    st.write('#### ROC curve of class `0`: ')
    plt.figure(figsize=(10, 6))
    st.pyplot(fc.ROC_curve_display(model, X_test_cv, y_test, 0))

elif choice == 'New Prediction':
    st.subheader('Select data')
    flag = False
    lines = None
    type = st.radio('Upload data or Input data?', options = ('Upload', 'Input'))
    if type == 'Upload':
        upload_file1 = st.file_uploader('Choose a file', type = ['txt', 'csv'])
        if upload_file1 is not None:
            lines = pd.read_csv(upload_file1, header = None)
            st.dataframe(lines)
            lines = lines[0]
            flag = True
    if type == 'Input':
        email = st.text_area(label = 'Input your content: ')
        if email != "":
            lines = np.array([email])
            flag = True
    if flag:
        st.write('Content: ')
        if len(lines) > 0:
            st.code(lines)
            x_new = cv.transform(lines)
            y_pred_new = model.predict(x_new)
            for x in y_pred_new:
                if x == 0:
                    st.code('Class 0 - This is Negative review.')
                else:
                    st.code('Class 1 - This is Positive review.')
            

    


    


    


    

