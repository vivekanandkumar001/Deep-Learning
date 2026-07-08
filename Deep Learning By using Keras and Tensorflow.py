import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)

df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")

df.shape
df.head(10)

df.target.value_counts(normalize = True , dropna = False)

categorical_variables = ['sex','cp' , 'fbs','restecg','exang','slope','ca','thal']
numerical_variables = ['age','trestbps','chol','thalach','oldpeak']

df = pd.get_dummies(df , columns = categorical_variables)
df = df.astype(int)

df.head()

test_df = df.sample(frac = 0.2 , random_state = 42)
train_df = df.drop(test_df.index)

train_df.shape

test_df.shape

means = train_df[numerical_variables].mean()
sd = train_df[numerical_variables].std()

means

train_df[numerical_variables] = (train_df[numerical_variables]-means)/sd
test_df [numerical_variables] = (test_df[numerical_variables]-means)/sd

train_df.head()

train = train_df.to_numpy()
test = test_df.to_numpy()

train_X = np.delete(train , 5 , axis = 1)
test_X = np.delete(test , 5 , axis = 1)

train_X.shape , test_X.shape

train_y = train[: , 5]
test_y = test[: ,5]

train_y.shape , test_y.shape

num_columns = train_X.shape[1]

#define Input Layer
input = keras.Input(shape = (num_columns,))
# feed the input vector to the hidden layers
#keep track . this doesnt affect the training etc.
h = keras.layers.Dense(16 , activation = 'relu' , name = 'hidden')(input)
#feed the output of the hidden layer to output layer
output = keras.layers.Dense(1 , activation = 'sigmoid' , name = 'output')(h)
#tell keras that this(input , output) pair is your model
model = keras.Model(input ,output)

"""The model.summary() command is a good way to get a quick overview of what you have defined."""

model.summary()

"""we can "visualize" the network graphically as well using keras plot_model function"""

keras.utils.plot_model(model , show_shapes = True)

model.compile(optimizer= 'adam',
              loss = 'binary_crossentropy' ,
              metrics = ['accuracy'])

"""Training"""

history = model.fit(train_X ,
                    train_y,
                    epochs = 300 ,
                    batch_size = 32,
                    verbose = 1,
                    validation_split = 0.2)

history_dict = history.history
history_dict.keys()

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values , "bo" , label = "Training loss")
plt.plot(epochs , val_loss_values , "b" , label = 'validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

"""there does to be overfitting

"""

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs,acc,"bo" , label = "training acc")
plt.plot(epochs , val_acc , 'b' , label = 'validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show

model.evaluate(test_X,test_y)
