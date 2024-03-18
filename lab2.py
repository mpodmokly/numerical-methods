import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_histogram(data_name):
    df_train[data_name].hist(bins=20, edgecolor="black")
    plt.title(f"Cancer {data_name} histogram")
    plt.xlabel("Value range")
    plt.ylabel("Count")
    plt.show()

def show_plot(data_name):
    df_train[data_name].plot(kind="line")
    plt.title(f"Cancer {data_name} plot")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()

df_train = pd.read_csv("breast-cancer-train.dat", header=None)
df_validate = pd.read_csv("breast-cancer-validate.dat", header=None)
columns = pd.read_csv("breast-cancer.labels", header=None)

df_train.columns = columns.iloc[:, 0].tolist()
df_validate.columns = columns.iloc[:, 0].tolist()

matrix_columns = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]
linear_train = df_train[matrix_columns].values
linear_validate = df_validate[matrix_columns].values

add1 = np.square(linear_train)
add2 = linear_train[:, 0] * linear_train[:, 1]
add2 = add2.reshape(-1, 1)
add3 = linear_train[:, 0] * linear_train[:, 2]
add3 = add3.reshape(-1, 1)
add4 = linear_train[:, 0] * linear_train[:, 3]
add4 = add4.reshape(-1, 1)
add5 = linear_train[:, 1] * linear_train[:, 2]
add5 = add5.reshape(-1, 1)
add6 = linear_train[:, 1] * linear_train[:, 3]
add6 = add6.reshape(-1, 1)
add7 = linear_train[:, 2] * linear_train[:, 3]
add7 = add7.reshape(-1, 1)
quadratic_train = np.hstack((linear_train, add1, add2, add3, add4, add5, add6, add7))

add1 = np.square(linear_validate)
add2 = linear_validate[:, 0] * linear_validate[:, 1]
add2 = add2.reshape(-1, 1)
add3 = linear_validate[:, 0] * linear_validate[:, 2]
add3 = add3.reshape(-1, 1)
add4 = linear_validate[:, 0] * linear_validate[:, 3]
add4 = add4.reshape(-1, 1)
add5 = linear_validate[:, 1] * linear_validate[:, 2]
add5 = add5.reshape(-1, 1)
add6 = linear_validate[:, 1] * linear_validate[:, 3]
add6 = add6.reshape(-1, 1)
add7 = linear_validate[:, 2] * linear_validate[:, 3]
add7 = add7.reshape(-1, 1)
quadratic_validate = np.hstack((linear_validate, add1, add2, add3, add4, add5, add6, add7))

linear_train = df_train.iloc[:, 2:].values
linear_validate = df_validate.iloc[:, 2:].values

b_train = np.where(df_train["Malignant/Benign"] == "M", 1, -1)
b_validate = np.where(df_validate["Malignant/Benign"] == "M", 1, -1)

#inv = np.linalg.inv(np.dot(linear_train.T, linear_train))
#w_linear = np.dot(np.dot(inv, linear_train.T), b_train)
#inv = np.linalg.inv(np.dot(quadratic_train.T, quadratic_train))
#w_quadratic = np.dot(np.dot(inv, quadratic_train.T), b_train)

ATA_linear = np.dot(linear_train.T, linear_train)
ATA_quadratic = np.dot(quadratic_train.T, quadratic_train)
w_linear = np.linalg.solve(ATA_linear, np.dot(linear_train.T, b_train))
w_quadratic = np.linalg.solve(ATA_quadratic, np.dot(quadratic_train.T, b_train))

cond_linear = np.linalg.cond(np.dot(linear_train.T, linear_train))
cond_quadratic = np.linalg.cond(np.dot(quadratic_train.T, quadratic_train))
print(f"cond linear method: {cond_linear}")
print(f"cond quadratic method: {cond_quadratic}\n")

p_linear = np.dot(linear_validate, w_linear)
p_quadratic = np.dot(quadratic_validate, w_quadratic)

malignant_no = len(np.where(b_validate == 1)[0])
benign_no = len(np.where(b_validate == -1)[0])

malignant_linear = len(np.where((p_linear > 0) & (b_validate == 1))[0])
benign_linear = len(np.where((p_linear <= 0) & (b_validate == -1))[0])
malignant_quadratic = len(np.where((p_quadratic > 0) & (b_validate == 1))[0])
benign_quadratic = len(np.where((p_quadratic <= 0) & (b_validate == -1))[0])

print(f"Correct diagnosis of malignant (linear): {malignant_linear}/{malignant_no}")
print(f"Correct diagnosis of benign (linear): {benign_linear}/{benign_no}")
print(f"Correct diagnosis of malignant (quadratic): {malignant_quadratic}/{malignant_no}")
print(f"Correct diagnosis of benign (quadratic): {benign_quadratic}/{benign_no}")

accurancy_linear = (malignant_linear + benign_linear) / (malignant_no + benign_no)
accurancy_quadratic = (malignant_quadratic + benign_quadratic) / (malignant_no + benign_no)
print(f"Accurancy of linear method: {round(accurancy_linear * 100, 2)}%")
print(f"Accurancy of quadratic method: {round(accurancy_quadratic * 100, 2)}%")

data_name = "radius (mean)"
#show_histogram(data_name)
#show_plot(data_name)
