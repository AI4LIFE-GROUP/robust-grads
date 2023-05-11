from sklearn.linear_model import LogisticRegression


def logistic_reg_l2(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    return clf

def logistic_reg_l1(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state, penalty='l1', solver='liblinear').fit(X,y)
    return clf

def logistic_reg_none(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state, penalty='none', solver='saga').fit(X,y)
    return clf
