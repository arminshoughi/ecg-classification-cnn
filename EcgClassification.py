import timeit, pywt
import numpy as np
import scipy.io as spio
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# read data from database
start_process = timeit.default_timer()
read_data_start = timeit.default_timer()
classes = [
    0,  # F
    1,  # N
    2,  # S
    3,  # V
    4,  # Q
]
classes_name = ["F", "N", "S", "V", "Q"]
data = []
input_length = 280
samples = spio.loadmat("data/mitbih_aami.mat")
samples = samples["s2s_mitbih"]
values = samples[0]["seg_values"]

labels = samples[0]["seg_labels"]
num_annots = sum([item.shape[0] for item in values])

#  add all segments(beats) together
l_data = 0
for i, item in enumerate(values):
    for itm in item:
        if l_data == num_annots:
            break
        data.append(np.interp(itm[0], (itm[0].min(), itm[0].max()), (0, 1)))
        l_data = l_data + 1

#  add all labels together
l_labels = 0
t_labels = []
for i, item in enumerate(labels):
    if len(t_labels) == num_annots:
        break
    item = item[0]
    for label in item:
        if l_labels == num_annots:
            break
        t_labels.append(str(label))
        l_labels = l_labels + 1

del values

data = np.reshape(data, (-1, 280))
data = np.asarray(data)

# normalize labels
n_labels = []

for i, v in enumerate(t_labels):
    if v == "F":
        n_labels.append(0)
    elif v == "N":
        n_labels.append(1)
    elif v == "S":
        n_labels.append(2)
    elif v == "V":
        n_labels.append(3)
    elif v == "Q":
        n_labels.append(4)

annotation = np.asarray(n_labels)

read_data_end = timeit.default_timer()
print("read data duration : {} seconds".format(read_data_end - read_data_start))


def class_details(labels):
    details = np.zeros(np.shape(classes))
    for l in labels:
        for i, v in enumerate(classes):
            if v == l:
                details[i] += 1
    return details


details = class_details(annotation)
for i, v in enumerate(classes):
    print("{:^5} : {:^5}".format(classes_name[i], details[i]))

# for i,j in enumerate([17814,0,6,1905,3168]):
#      i+=1
#      plt.subplot(5,2,(2*i)-1)
#      plt.title(classes_name[annotation[j]])
#      plt.plot(data[j],'r')

wt_start = timeit.default_timer()
print("wavelet transform process ...")
data1, data2 = pywt.dwt(data, "db2")
data = data1
# data1,data2=pywt.dwt(data,'db2')
# data=data1
wt_end = timeit.default_timer()
print("done")
print("Wavelet transform duration : {} seconds".format(wt_end - wt_start))
input_length = np.shape(data2)[1]
print("input shape {}".format(input_length))
print("data1 {}".format(np.shape(data1)))
print("data2 {}".format(np.shape(data2)))

# for d in data:
#    d = np.reshape(d,(input_length,1))

# for i,j in enumerate([17814,0,6,1905,3168]):
#      plt.subplot(2,5,i+6)
#      plt.title(classes_name[annotation[j]])
#      plt.plot(data[j],'b')
# plt.show()
#
# exit(1)

# split data to train and test 9-1 ...
print("split data to train and test 9-1 ...")
inter_x_train, inter_x_test, inter_y_train, inter_y_test = train_test_split(
    data, annotation, test_size=0.1, train_size=0.9, shuffle=True
)
print("train data")
details = class_details(inter_y_train)
for i, v in enumerate(classes):
    print("{:^5} : {:^5}".format(classes_name[i], details[i]))

print("test data")
details = class_details(inter_y_test)
for i, v in enumerate(classes):
    print("{:^5} : {:^5}".format(classes_name[i], details[i]))

# over sampling process ...
oversampling_start = timeit.default_timer()
print("over sampling proccess ...")
sm = SMOTE(sampling_strategy={0: 2000, 2: 5000})
oversampling_end = timeit.default_timer()
print(
    "oversampling duration : {} seconds".format(oversampling_end - oversampling_start)
)
inter_x_train, inter_y_train = sm.fit_resample(inter_x_train, inter_y_train)
print("train data")
details = class_details(inter_y_train)
for i, v in enumerate(classes):
    print("{:^5} : {:^5}".format(classes_name[i], details[i]))

print("test data")
details = class_details(inter_y_test)
for i, v in enumerate(classes):
    print("{:^5} : {:^5}".format(classes_name[i], details[i]))

print(
    "shape of : xtrain {} ytrain {} xtest{} ytest{}".format(
        np.shape(inter_x_train),
        np.shape(inter_y_train),
        np.shape(inter_x_test),
        np.shape(inter_y_test),
    )
)

# resize samples to prefix size
inter_x_train = np.reshape(inter_x_train, [-1, input_length, 1])
inter_x_test = np.reshape(inter_x_test, [-1, input_length, 1])
inter_y_train = np.reshape(inter_y_train, [-1, 1])
inter_y_test = np.reshape(inter_y_test, [-1, 1])

print("input length > {}".format(input_length))

# build CNN Model
import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Conv1D(
        filters=100,
        input_shape=(input_length, 1),
        kernel_size=(5),
        strides=(2),
        activation="relu",
    )
)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.MaxPool1D(pool_size=(3)))

model.add(
    tf.keras.layers.Conv1D(filters=90, kernel_size=(5), strides=(2), activation="relu")
)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.MaxPool1D(pool_size=(3)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(30, activation="relu"))
model.add(tf.keras.layers.Dense(5, activation="softmax"))

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
    ],
)

training_start = timeit.default_timer()
history = model.fit(
    inter_x_train,
    inter_y_train,
    validation_split=0.1,
    epochs=1,
    batch_size=20,
)
training_end = timeit.default_timer()
print("Training duration : {} seconds".format(training_end - training_start))

# plt.plot(history.history["accuracy"], "r")
# plt.plot(history.history["val_accuracy"], "b")
# plt.title("model accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.show()

loss, acc = model.evaluate(inter_x_test, inter_y_test, verbose=2)

print("validation loss : {} \naccuracy : {}".format(loss, acc))
print(30 * "#")

prediction_start = timeit.default_timer()
predictions = model.predict(inter_x_test)
prediction_end = timeit.default_timer()
print("Prediction duration : {} seconds".format(prediction_end - prediction_start))

matrix = np.zeros((10, 2))

for i, x in enumerate(inter_x_test):
    predicted_value = np.argmax(predictions[i], axis=0)
    correct_value = inter_y_test[i]
    # Class F
    if predicted_value == 0 and correct_value == 0:
        matrix[0, 0] = matrix[0, 0] + 1

    elif predicted_value == 0 and correct_value != 0:
        matrix[1, 0] = matrix[1, 0] + 1

    elif correct_value == 0 and predicted_value != 0:
        matrix[0, 1] = matrix[0, 1] + 1

    elif correct_value != 0 and predicted_value != 0:
        matrix[1, 1] = matrix[1, 1] + 1

    # Class N
    if predicted_value == 1 and correct_value == 1:
        matrix[2, 0] = matrix[2, 0] + 1

    elif predicted_value == 1 and correct_value != 1:
        matrix[3, 0] = matrix[3, 0] + 1

    elif correct_value == 1 and predicted_value != 1:
        matrix[2, 1] = matrix[2, 1] + 1

    elif correct_value != 1 and predicted_value != 1:
        matrix[3, 1] = matrix[3, 1] + 1

    # Class S
    if predicted_value == 2 and correct_value == 2:
        matrix[4, 0] = matrix[4, 0] + 1

    elif predicted_value == 2 and correct_value != 2:
        matrix[5, 0] = matrix[5, 0] + 1

    elif correct_value == 2 and predicted_value != 2:
        matrix[4, 1] = matrix[4, 1] + 1

    elif correct_value != 2 and predicted_value != 2:
        matrix[5, 1] = matrix[5, 1] + 1

    # Class V
    if predicted_value == 3 and correct_value == 3:
        matrix[6, 0] = matrix[6, 0] + 1

    elif predicted_value == 3 and correct_value != 3:
        matrix[7, 0] = matrix[7, 0] + 1

    elif correct_value == 3 and predicted_value != 3:
        matrix[6, 1] = matrix[6, 1] + 1

    elif correct_value != 3 and predicted_value != 3:
        matrix[7, 1] = matrix[7, 1] + 1

    # Class Q
    if predicted_value == 4 and correct_value == 4:
        matrix[8, 0] = matrix[8, 0] + 1

    elif predicted_value == 4 and correct_value != 4:
        matrix[9, 0] = matrix[9, 0] + 1

    elif correct_value == 4 and predicted_value != 4:
        matrix[8, 1] = matrix[8, 1] + 1

    elif correct_value != 4 and predicted_value != 4:
        matrix[9, 1] = matrix[9, 1] + 1

print(matrix)

matrix2 = np.zeros((5, 5))
for i, x in enumerate(inter_x_test):
    for index in range(0, 5):
        if np.argmax(predictions[i], axis=0) == index:
            if inter_y_test[i] == 0:
                matrix2[index, 0] = matrix2[index, 0] + 1
            elif inter_y_test[i] == 1:
                matrix2[index, 1] = matrix2[index, 1] + 1
            elif inter_y_test[i] == 2:
                matrix2[index, 2] = matrix2[index, 2] + 1
            elif inter_y_test[i] == 3:
                matrix2[index, 3] = matrix2[index, 3] + 1
            elif inter_y_test[i] == 4:
                matrix2[index, 4] = matrix2[index, 4] + 1

print(matrix2)

print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10}\t{:^10}\t{:^10}\t{:^10}".format(
        "Class", "F", "N", "S", "V", "Q", "acc", "Sen", "Spac", "PPV"
    )
)
print(112 * "*")
print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}".format(
        "F",
        matrix2[0, 0],
        matrix2[0, 1],
        matrix2[0, 2],
        matrix2[0, 3],
        matrix2[0, 4],
        (
                (matrix[0, 0] + matrix[1, 1])
                / (matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])
        )
        * 100,
        (matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])) * 100,
        (matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])) * 100,
        (matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])) * 100,
    )
)
print(112 * "*")
print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}".format(
        "N",
        matrix2[1, 0],
        matrix2[1, 1],
        matrix2[1, 2],
        matrix2[1, 3],
        matrix2[1, 4],
        (
                (matrix[2, 0] + matrix[3, 1])
                / (matrix[2, 0] + matrix[2, 1] + matrix[3, 0] + matrix[3, 1])
        )
        * 100,
        (matrix[2, 0] / (matrix[2, 0] + matrix[3, 0])) * 100,
        (matrix[3, 1] / (matrix[3, 1] + matrix[2, 1])) * 100,
        (matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])) * 100,
    )
)
print(112 * "*")
print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}".format(
        "S",
        matrix2[2, 0],
        matrix2[2, 1],
        matrix2[2, 2],
        matrix2[2, 3],
        matrix2[2, 4],
        (
                (matrix[4, 0] + matrix[5, 1])
                / (matrix[4, 0] + matrix[4, 1] + matrix[5, 0] + matrix[5, 1])
        )
        * 100,
        (matrix[4, 0] / (matrix[4, 0] + matrix[5, 0])) * 100,
        (matrix[5, 1] / (matrix[5, 1] + matrix[4, 1])) * 100,
        (matrix[4, 0] / (matrix[4, 0] + matrix[4, 1])) * 100,
    )
)
print(112 * "*")
print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}".format(
        "V",
        matrix2[3, 0],
        matrix2[3, 1],
        matrix2[3, 2],
        matrix2[3, 3],
        matrix2[3, 4],
        (
                (matrix[6, 0] + matrix[7, 1])
                / (matrix[6, 0] + matrix[6, 1] + matrix[7, 0] + matrix[7, 1])
        )
        * 100,
        (matrix[6, 0] / (matrix[6, 0] + matrix[7, 0])) * 100,
        (matrix[7, 1] / (matrix[7, 1] + matrix[6, 1])) * 100,
        (matrix[6, 0] / (matrix[6, 0] + matrix[6, 1])) * 100,
    )
)
print(112 * "*")
print(
    "\n{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}\t{:^10.2f}".format(
        "Q",
        matrix2[4, 0],
        matrix2[4, 1],
        matrix2[4, 2],
        matrix2[4, 3],
        matrix2[4, 4],
        (
                (matrix[8, 0] + matrix[9, 1])
                / (matrix[8, 0] + matrix[8, 1] + matrix[9, 0] + matrix[9, 1])
        )
        * 100,
        (matrix[8, 0] / (matrix[8, 0] + matrix[9, 0])) * 100,
        (matrix[9, 1] / (matrix[9, 1] + matrix[8, 1])) * 100,
        (matrix[8, 0] / (matrix[8, 0] + matrix[8, 1])) * 100,
    )
)
print(112 * "*")