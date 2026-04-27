import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X = digits.data        # (1797, 64)
y = digits.target      # 10 classes (0-9)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=200,
    random_state=42
)

sgd_model.fit(X_train, y_train)

sgd_loss = sgd_model.loss_curve_
sgd_pred = sgd_model.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_pred)



adam_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42
)

adam_model.fit(X_train, y_train)

adam_loss = adam_model.loss_curve_
adam_pred = adam_model.predict(X_test)
adam_accuracy = accuracy_score(y_test, adam_pred)



print("\n========== Accuracy Comparison ==========")
print("SGD Accuracy  :", round(sgd_accuracy * 100, 2), "%")
print("Adam Accuracy :", round(adam_accuracy * 100, 2), "%")



plt.figure()
plt.plot(sgd_loss, label="SGD")
plt.plot(adam_loss, label="Adam")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iteration (SGD vs Adam)")
plt.legend()
plt.show()

cm_sgd = confusion_matrix(y_test, sgd_pred)

plt.figure()
disp_sgd = ConfusionMatrixDisplay(confusion_matrix=cm_sgd)
disp_sgd.plot()
plt.title("Confusion Matrix - SGD")
plt.show()

cm_adam = confusion_matrix(y_test, adam_pred)

plt.figure()
disp_adam = ConfusionMatrixDisplay(confusion_matrix=cm_adam)
disp_adam.plot()
plt.title("Confusion Matrix - Adam")
plt.show()



print("\n========== Final Comparison ==========")
if adam_accuracy > sgd_accuracy:
    print("Adam performs better than SGD on this dataset.")
elif sgd_accuracy > adam_accuracy:
    print("SGD performs better than Adam on this dataset.")
else:
    print("Both optimizers perform equally.")

print("SGD Final Loss :", sgd_loss[-1])
print("Adam Final Loss:", adam_loss[-1])