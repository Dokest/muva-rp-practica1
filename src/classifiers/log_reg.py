from sklearn.linear_model import LogisticRegression


def log_reg_classifier():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    """
    return {
        "classifier": [LogisticRegression()],
        "classifier__fit_intercept": [True, False],
        "classifier__max_iter": [10000],
        # "classifier__dual": [True, False],
    }
