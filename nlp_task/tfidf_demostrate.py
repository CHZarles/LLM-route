import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# this is a very toy example,
# do not try this at home unless you want to understand the usage differences 

# Usage 1: CountVectorizer
# 这里相当于5个文档，每个文档有不同的词汇
docs=["the house had a tiny little mouse", 
"the cat saw the mouse", 
"the mouse ran away from the house", 
"the cat finally ate the mouse", 
"the end of the mouse story" ]


# stage 1 , calculate TF
#instantiate CountVectorizer() 
cv=CountVectorizer() 

# this steps generates word counts for the words in your docs 
# 这里实际上是生成了一个词频矩阵
word_count_vector=cv.fit_transform(docs)
print(word_count_vector.shape)
print(cv.vocabulary_)
print(word_count_vector.toarray())



# stage 2, calculate IDF

"""
Attention!
We could have actually used word_count_vector from above.
However, in practice, you may be computing tf-idf scores on a set of new unseen documents.
When you do that, you will first have to do cv.
transform(your_new_docs) to generate the matrix of word counts.
"""
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

# count matrix 
count_vector=cv.transform(docs) 
# tf-idf scores 
tf_idf_vector=tfidf_transformer.transform(word_count_vector)

print(tf_idf_vector.toarray())



# Usage 2: TfidfVectorizer
# TfidfVectorizer = CountVectorizer + TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# settings that you use for count vectorizer will go here 
tfidf_vectorizer=TfidfVectorizer(use_idf=True) 

# just send in all your docs here 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)

print(tfidf_vectorizer_vectors.toarray())

"""
这样用也行
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

# just send in all your docs here
fitted_vectorizer=tfidf_vectorizer.fit(docs)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs)

"""
