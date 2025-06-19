from sklearn.linear_model import LogisticRegression # type: ignore 
from sklearn.model_selection import cross_val_score # type: ignore 

def train_model(X_train, y_train):
    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    score = cross_val_score(log_clf, X_train, y_train, cv=3)
    print("Cross-validation score:", score.mean())
    log_clf.fit(X_train, y_train)
    return log_clf
