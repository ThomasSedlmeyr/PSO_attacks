import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from src.utils.utils import read_config

if __name__ == "__main__":
    config = read_config()
    path = config.path_data + "cardio/" + "cardio_cat.csv"
    df = pd.read_csv(path, delimiter=',')
    print(df.shape)
    #df.info()
    X = df.drop(["cardio"], axis=1)
    #X.drop(["id"], axis=1, inplace=True)
    print(X.info())
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_rnf = RandomForestClassifier()
    parametrs_rnf = {'n_estimators': [15, 20, 30, 50], 'max_depth': [5, 10, 15, 20, 30]}
    grid_forest = GridSearchCV(clf_rnf, parametrs_rnf, cv=6, n_jobs=-1)
    grid_forest.fit(X_train, y_train)

    best_model_rnf = grid_forest.best_estimator_
    y_pred_rnf = best_model_rnf.predict(X_test)

    ac_rnf = accuracy_score(y_test, y_pred_rnf)
    print("Accuracy score for model " f'{best_model_rnf} : ', ac_rnf, '\n')
    cr_rnf = classification_report(y_test, y_pred_rnf)
    print("classification_report for model " f'{best_model_rnf} : \n', cr_rnf)