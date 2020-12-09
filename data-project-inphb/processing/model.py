# Importation de package
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

# Partitiionnement en Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# On utilise donc la moyenne de plusieurs validation croisées pour augmenter
# la significativité de la validation
def compute_score(clf, X, y, cv=5):
    """compute score in a classification modelisation.
       clf: classifier
       X: features
       y: target
    """
    xval = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (xval.mean(), xval.std() * 2))
    return xval


# ce score est proche de la moyenne de la vingtaine de tests de validation réalisée plus haut.
# ce qui valide la pertinence de cette stratégie de croos validation.

#ypredproba = Reg_Log.predict_proba(X_test)[:,1]
def My_model ( X, y, size, RdomState = 42) :
    #X, y
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=size, 
                                                       random_state=RdomState )
    model = LogisticRegression(random_state= RdomState)
    model.fit(X_train, y_train)
    # Run the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    metric = metrics.classification_report(y_test, ypred)
    
    return {"y_test": y_test, "prediction": y_pred, "proba":y_prob,
           "score_train": score_train, "score_test": score_test,
           "model": model, "metric": print(metric)}
  