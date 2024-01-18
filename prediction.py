import joblib
def predict(data):
    clf = joblib.load("Breastcnacer_rf_model.sav")
    return clf.predict(data)
