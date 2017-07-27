# Connect 4 machine learning model
from keras.models import Sequential
from keras.layers import Dense
import numpy

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load connect 4 dataset
dataset = numpy.genfromtxt("connect-4.csv", dtype='str', delimiter=",")
# split into input (X) and output (Y) variables
preX = dataset[:,0:42]
print preX
preY = dataset[:,42]
print preY

X = numpy.zeros(preX.shape)
Y = numpy.zeros(preY.shape)

for i, row in enumerate(preX):
    for j, col in enumerate(row):
        if col == 'x':
            X[i,j] = 1.0
        if col == 'o':
            X[i,j] = -1.0
        if col == 'b':
            X[i,j] = 0.0

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=42, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs = 10, batch_size=5,verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model.save('connect_four_model.h5')

# accuracy of 78.6%