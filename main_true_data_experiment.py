import src.data_loading as data_loading
from src.logistic_regression import LogisticRegression
import src.utils as utils
import time


if __name__ == "__main__":
    # Load the true dataset
    X_train, y_train, X_test, y_test = data_loading.load_gisette_local()

    model = LogisticRegression()

    start_time = time.time()
    # Initialize the Logistic Regression model
    model.fit(X_train, y_train, method='analytic')
    end_time = time.time()
    print(f"Training time for Logistic Regression on true data: {end_time - start_time:.4f} seconds")

    # Evaluate the model
    accuracy_train = model.evaluate(X_train, y_train)
    print(f"Model accuracy on training data: {accuracy_train:.2f}")

    accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy on true data: {accuracy:.2f}")

    loss_curve = model.get_loss_curve()
    # Plot the loss curve
    utils.plot_loss_curve(loss_curve)
