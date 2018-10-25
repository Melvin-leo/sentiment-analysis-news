##Sentiment Analysis of News with Keras (Machine Learning)

This repository is the implementation of Keras for general sentiment analysis on news.

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
In this repository we will be using Keras on TensorFlow Backend. Refer to [Keras](https://keras.io/) for more information on Keras. 

Recurrent neural network is used for this research. Hence, LSTM will be used.
Refer to `sentiment-analysis-model`'s notebook.

##Data used
Due to the lack of actual news with sentiments, tweets from twitter are used. However much pre-processing is needed so that the dataset are more relevant and accurate.

Furthermore negative news can be added manually to csv file that can be input into the model.
Steps to combine tweets and news from csv to another can be found in the `combine_tweet_news` notebook.


##Prepocess twitter data
The steps to pre-process twitter data according to the datasets can be found in `preprocess_twitter_data` notebook.


#Models and datasets used
Currently there are 4 models that produce relatively good results. These 3 models are built by 3 different datasets that contains a mixture of tweets and news.
Tweets from twitter are used to generate common words that are already classified with sentiments. There are a couple of datasets available online that are ready to be used and hence we will be using them.
The tweets will be pre-processed and saved into csv file to be combined with other datas.

For `text_negative.csv` and `self_negative_news.csv` they are made by collecting some news data and are all negative sentiments. However they are the focus and what we are trying to filter.

There are individual csv files for each models in this repository.

For model 5, dataset that is used consists of :
```
#model 5
# 2363 positive airline
# 5000 negative airline
# 2193 general positive
# 3000 general negative
# 3444 general positive
# 2000 general positive
# self write 200 negative
```
The performance is decent however some test cases are wrong. Not recommended.

For model 6, dataset that is used consists of:
```
#model 6
# text negative 1820
# self write 200 negative
# 4980 airplane negative
# 3000 general negative
# 5461 airplane positive
# 4539 general positive
```
To train the model, dataset was split by `scikitlearn` `train_test_split` where only 0.66 of the data was randomly picked for training.
The perfomance on model 6 is rather accurate for the tests cases tried.
Not recommended as `train_test_split` randomised the input into the model.

For model 7, dataset that is used consists of:
```
#model 7
# text negative 1820
# self write 200 negative
# 4950 airplane negative 
# 3000 general negative
# 5461 airplane positive
# 4539 general positive
```
Similar dataset was used for model 7. However, `train_test_split` was not used. This is to fully use the dataset that is created for the model and to test the model with `news_tests.csv`.
Relatively accurate.

For model 9, dataset that is used consists of:
```
#model 9
#2394 negative news and tweets
#3000 airline negative tweets
#2506 negative normal tweets
#4000 airline neutral/positive tweets
#4000 positive normal tweets
```
New dataset is introduced for model 9. This dataset have been further processed to remove some of the unnecessary words that are repeated more frequently in some of the data.
`train_test_split` is once again not used for better testing and training.


**Tried increasing the number of positive data, added 5000 more positive data than negative data but results deteriorated, might be due to the increase in weight for non relevant words thatreduced on the impact of weight of negative sentiments.
##Converting the data into sequences to be fed into the model
```
data['text'] = data['text'].apply(lambda x: str(x))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)
    
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

```
`max_features` can be manipulated to increase the number of words to be included in the vector for the model.

##Initialising Keras model
```
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
```
The reason why `categorical_crossentropy` is chosen is because this is a multi class classification task. Hence we will prefer to use `categorical_crossentropy`.

The optimizer controls the learning rate. ‘adam’ is used as the optmizer. Adam is generally a good optimizer to use for many cases. The adam optimizer adjusts the learning rate throughout training.

`embed_dim`,`batch_size` and `lstm_out` can also be changed to improve accuracy of the model.

