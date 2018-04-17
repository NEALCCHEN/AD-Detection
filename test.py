
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
'''
# 选取参与分析的文本类别
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
'''
# 从硬盘获取原始数据
movie=load_files("D:\Python\Movie",
        load_content = True,
        encoding="latin1",
        decode_error="strict",
        shuffle=True, random_state=42)
# 统计词语出现次数
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(movie.data)
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# 打印特征矩阵规格
f=open("D:\Python\Movie\Result","w")
print("Feature Matrix Transformed from Text File ")
print(X_train_tfidf.toarray()) 
print(X_train_tfidf.shape)
f.close

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, movie.target)
print("The information about the Classifier：")
print(clf)

# 预测用的新字符串，你可以将其替换为任意英文句子
docs_new = ["nb"]
# 字符串处理
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# 进行预测
predicted = clf.predict(X_new_tfidf)

# 打印预测结果
for doc, category in zip(docs_new, predicted):
    print("%r => %s" % (doc, movie.target_names[category]))
    
from sklearn.pipeline import Pipeline
# 建立Pipeline
text_clf = Pipeline([("vect", CountVectorizer()),
                     ("tfidf", TfidfTransformer()),
                     ("clf", MultinomialNB()),
])
# 训练分类器
    # twenty_train--movie
text_clf = text_clf.fit(movie.data, movie.target)
# 打印分类器信息
print(text_clf)    

# 获取测试数据
movie_test=load_files('D:\Python\Movie_test',
                        load_content = True, 
                        encoding='latin1',
                        decode_error='strict',
                        shuffle=True, random_state=42)
docs_test = movie_test.data
# 使用测试数据进行分类预测
predicted = text_clf.predict(docs_test)
# 计算预测结果的准确率
print("The accuracy rat：")
print(np.mean(predicted == movie_test.target))

from sklearn import metrics
print("The Performance of Classification ：")
print(metrics.classification_report(movie_test.target, predicted,
    target_names=movie_test.target_names))
print("Confusion Matrix：")
metrics.confusion_matrix(movie_test.target, predicted)

