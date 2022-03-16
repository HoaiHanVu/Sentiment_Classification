import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import import_ipynb
from Lib import Functions as fc

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

vocab_size = 1200
max_length = 150
embedding_dim = 64
oov_tok = '<OOV>'
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

train_sequence = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequence, maxlen = max_length, padding = padding_type, truncating = trunc_type)

valid_sequence = tokenizer.texts_to_sequences(X_test)
valid_padded = pad_sequences(valid_sequence, maxlen = max_length, padding = padding_type, truncating = trunc_type)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 3. Build model
# tf.keras.backend.clear_session()
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
#     tf.keras.layers.Conv1D(32, 3, activation = 'relu'),
#     tf.keras.layers.MaxPooling1D(pool_size = 2, strides = 1),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(32, activation = 'elu', kernel_initializer = 'he_normal'),
#     tf.keras.layers.Dense(24, activation = 'elu', kernel_initializer = 'he_normal'),
#     tf.keras.layers.Dense(16, activation = 'elu', kernel_initializer = 'he_normal'),
#     tf.keras.layers.Dense(1, activation = 'sigmoid')
# ])

# model.compile(loss = 'binary_crossentropy',
#               optimizer = 'adam',
#               metrics = ['accuracy'])

# history = model.fit(train_padded, y_train,
#                     validation_data = (valid_padded, y_test),
#                     epochs = 150,
#                     verbose = 2)
# df_his = pd.DataFrame(history.history)

# yhat = model.predict(valid_padded)
# yhat = np.round(yhat).reshape(-1)

# cr = classification_report(y_test, yhat)

# 4. Load model & evaluate:

classifier = load_model('Foody_review_analysis.h5')
stringlist = []
classifier.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

yhat = classifier.predict(valid_padded)
yhat = np.round(yhat).reshape(-1)

df_his = pd.read_csv('model_history.csv')
train_eval = classifier.evaluate(train_padded, y_train)
test_eval = classifier.evaluate(valid_padded, y_test)

report = classification_report(y_test, yhat)
conf_matrix = confusion_matrix(y_test, yhat)

def predict_review(text):
    new_sequence = tokenizer.texts_to_sequences(text)
    new_padded = pad_sequences(new_sequence,
                               padding = padding_type,
                               truncating = trunc_type,
                               maxlen = max_length)
    return new_padded



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
    st.code(short_model_summary)
    st.write('#### 2. Model Evaluation: ')
    st.write('##### Training history: ')
    st.dataframe(df_his.iloc[:, 1:].tail(5))
    st.write('##### Train Score: ')
    st.code(' - Accuracy: ' + str(np.round(train_eval[1], 3)))
    st.code(' - Loss: ' + str(np.round(train_eval[0], 3)))
    st.write('##### Valid Score: ')
    st.code(' - Accuracy: ' + str(np.round(test_eval[1], 3)))
    st.code(' - Loss: ' + str(np.round(test_eval[0], 3)))
    st.write('##### Classification Report: ')
    st.code(report)
    st.write('##### Confusion Matrix: ')
    st.code(conf_matrix)

    st.write('#### Loss value during training.')
    plt.figure(figsize=(10, 6))
    fig3 = df_his.loc[:, ['loss', 'val_loss']].plot(figsize=(10, 6))
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.grid(True)
    st.pyplot(fig3.figure)

    st.write('#### Accuracy score during training.')
    plt.figure(figsize=(10, 6))
    fig3 = df_his.loc[:, ['accuracy', 'val_accuracy']].plot(figsize=(10, 6))
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.grid(True)
    st.pyplot(fig3.figure)

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
            x_new = predict_review(lines)
            y_pred_new = np.argmax(classifier.predict(x_new), axis = -1)
            for x in y_pred_new:
                if x == 0:
                    st.code('Class 0 - This is Negative review.')
                else:
                    st.code('Class 1 - This is Positive review.')
            

    


    

