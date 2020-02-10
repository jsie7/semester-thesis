"""Python file containing arguments for turbidity classification

This file contains functions returning either a predictor or scaler
for the turbidity prediction.

"""

# third party modules
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def predictor():
    """Function returning a predictor instance.

    This function is used as an argument for the predict_turbidity function
    in the turbidityprediction module.

    Args:
        None.

    Returns:
        Predictor instance.

    """
    return MLPClassifier(hidden_layer_sizes=(50, 50, 50),
                         activation='logistic', random_state=21)


def scaler():
    """Function returning a scaler instance.

    This function is used as an argument for the predict_turbidity function
    in the turbidityprediction module.

    Args:
        None.

    Returns:
        Scaler instance.

    """
    return StandardScaler()
