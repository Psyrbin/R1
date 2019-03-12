from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import numpy as np

model = Sequential()

model.add(Dense(units=135, activation='relu', input_dim=30002, kernel_regularizer=regularizers.l2(0.08)))
model.add(Dense(units=75, activation='relu', input_dim=30002, kernel_regularizer=regularizers.l2(0.08)))
model.add(Dense(units=35, activation='relu', input_dim=30002, kernel_regularizer=regularizers.l2(0.08)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

bows = np.load('bows.npy')
targets = np.load('targets.npy').reshape((-1,1))


y = targets[:40000]
delete = np.random.rand(y.shape[0])
del_arr = []
for idx, elem in enumerate(delete):
    if elem < 0.98 and y[idx] == 1:
        del_arr.append(idx)
x_train = np.delete(bows, del_arr, 0)
y_train = np.delete(y, del_arr, 0)

model.fit(x_train, y_train, epochs=50, batch_size=1000, validation_split=0.0)

x_test = np.load('bows5.npy')
y_test = targets[40000:]

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# print(loss_and_metrics)


res = model.predict(x_train, batch_size=128)
res = res > 0.5
print('train accuracy: ', np.sum(np.equal(res, y_train)) / res.shape[0])
print('train neg accuracy: ', np.sum(np.equal(res, y_train) * (np.equal(y_train, np.zeros(y_train.shape[0])))) / np.sum(np.equal(y_train, np.zeros(y_train.shape[0]))))

res = model.predict(bows, batch_size=128)
res = res > 0.5
print('all accuracy: ', np.sum(np.equal(res, targets[:40000])) / 40000)
#print('all neg accuracy: ', np.sum(np.equal(res, targets[:40000]) * (np.equal(targets[:40000], np.zeros(40000)))) / np.sum(np.equal(targets[:40000], np.zeros(40000))))


res = model.predict(x_test, batch_size=128)
res = res > 0.5
print('test accuracy: ', np.sum(np.equal(res, targets[40000:])) / res.shape[0])
print('test neg accuracy: ', np.sum(np.equal(res, targets[40000:]) * (np.equal(targets[40000:], np.zeros(10000)))) / np.sum(np.equal(targets[40000:], np.zeros(10000))))