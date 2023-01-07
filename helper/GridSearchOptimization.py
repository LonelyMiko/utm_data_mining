from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


def optimize(model, X, y):
    # Evaluate model configurations
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] = ['l2', 'elasticnet']
    space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    # define grid
    grid = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    grid_result = grid.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
