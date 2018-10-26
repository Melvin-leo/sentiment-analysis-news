##Sentiment Analysis of News with Keras (Machine Learning)

This repository is the implementation of Keras for general sentiment analysis on news. For this repository we are only classifying news to positive or negative sentiments.

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
In this repository we will be using Keras on TensorFlow Backend. Refer to [Keras](https://keras.io/) for more information on Keras. 

Recurrent neural network is used for this research. Hence, LSTM will be used.
Refer to `sentiment-analysis-model9` notebook.

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

Refer to model 9 for the most updated and accurate model.

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
This is the best model so far in the research with accuracy up to 90 percent for both negative enad positive.

**Tried increasing the number of positive data, added 5000 more positive data than negative data but results deteriorated, might be due to the increase in weight for non relevant words thatreduced on the impact of weight of negative sentiments.
##Converting the data into sequences to be fed into the model
```
data['text'] = data['text'].apply(lambda x: str(x))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)
    
max_features = 3000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

```
`max_features` can be manipulated to increase the number of words to be included in the vector for the model.

`fit_on_texts` on tokenizer will fit the most common words that will be added into the features that is limited by `max_features`.

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


##Getting the Y axis(sentiments)
```
Y_train = pd.get_dummies(data['sentiment']).values
Y_test = pd.get_dummies(test_data['sentiment']).values
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
```

This step is vital to get the values for sentiments as arrays to be used for both training and testing.


##Training the model
```
batch_size = 32
model.fit(X_train, Y_train, epochs = 12, batch_size=batch_size, verbose = 2)
```
In Keras, `fit` will train the model easily. `batch_size` can altered to test accuracy as well.

##Testing the model with text or news to be input for sentiment analysis
```
txt = ['major delay in traffic due to accident']
#vectorizing the tweet by the pre-fitted tokenizer instance
txt = tokenizer.texts_to_sequences(txt)
#padding the tweet to have exactly the same shape as `embedding_2` input
txt = pad_sequences(txt, maxlen=222, dtype='int32', value=0)
# print(twt)
sentiment = model.predict(txt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
```
Make use of the `tokenizer` that was created earlier to convert the text to sequences and after padding the sequence, `model.predict` will predict the sentiment.

*`maxlen` is important as it have to be the same as the shape of `X_train` that was obtain earlier.

##Saving and loading the model
In Keras, model can be saved into h5 file. Make sure that `h5py` is installed and simple just call `model.save('<filename>.h5')`. 

```
from keras.models import load_model
model6 = load_model('my_model6.h5')
```
After the model is saved into the h5 file, will load the model and model is ready for both prediction or training.
##Things to note
The issue with the current models provided is that the datasets used are not fully related to the sentiment analysis of news. Much of the data used are twitter's tweets that contained some of the words that are not relevant and useless. The quality of the data can be further improved with more actual data of news, positive or negative.

