import pandas as pd;
import keras;
import tensorflow;
from sklearn.preprocessing import LabelEncoder;

train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv') 

def encode(train, test): 
    label_encoder = LabelEncoder().fit(train.AN) 
    labels = label_encoder.transform(train.AN) 
    classes = list(label_encoder.classes_) 

    train = train.drop(['cl'], axis=1) 

    return train, labels, test, classes 

train, labels, test, classes = encode(train, test) 

# standardize train features 
scaler = StandardScaler().fit(train.values) 
scaled_train = scaler.transform(train.values) 

# split train data into train and validation 
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23) 
for train_index, valid_index in sss.split(scaled_train, labels): 
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index] 
    y_train, y_valid = labels[train_index], labels[valid_index] 


nb_features = 1 # number of features per features type (shape, texture, margin) 
nb_class = len(classes) 

# reshape train data 
X_train_r = np.zeros((len(X_train), nb_features, 4)) 
X_train_r[:, :, 0] = X_train[:, 0:1] 
X_train_r[:, :, 1] = X_train[:, 1:2] 
X_train_r[:, :, 2] = X_train[:, 2:3] 
X_train_r[:, :, 3] = X_train[:, 3:4] 

# reshape validation data 
X_valid_r = np.zeros((len(X_valid), nb_features, 4)) 
X_valid_r[:, :, 0] = X_valid[:, 0:1] 
X_valid_r[:, :, 1] = X_valid[:, 1:2] 
X_valid_r[:, :, 2] = X_valid[:, 2:3] 
X_valid_r[:, :, 3] = X_valid[:, 3:4] 

## 



## 


# Keras model with one Convolution1D layer 
# unfortunately more number of covnolutional layers, filters and filters lenght 
# don't give better accuracy 
model = Sequential() 
model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(nb_features, 4))) 
model.add(Activation('relu')) 
model.add(Flatten()) 
model.add(Dropout(v[3])) 
model.add(Dense(2048, activation='relu')) 
model.add(Dense(1024, activation='relu')) 
model.add(Dense(nb_class)) 
model.add(Activation('softmax')) 


y_train = np_utils.to_categorical(y_train, nb_class) 
y_valid = np_utils.to_categorical(y_valid, nb_class) 






sgd = SGD(lr=v[0], nesterov=True, decay=v[1], momentum=v[2]) 
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy']) 

nb_epoch = 15 
model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16) 
# predicted_val=model.predict(X_train_r) 
model.get_weights() 

scores = model.evaluate(X_train_r, y_train, verbose=0)
