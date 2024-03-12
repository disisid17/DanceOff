from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import argparse


"""
This file will run through a number of scikit learn models on the the training data
in training.csv.  This training data was collected through running:
"""

model_name = 'level1_pose_model'


def get_data(file_name):
    """
    read training.csv and return the X,y as series
    :return: X - the data representing the road view
             y - what turn value
    """
    df = pd.read_csv(f'{file_name}', header=None)
    # print(df.head())
    X = df.loc[:, 1:]
    y = df.loc[:, 0]
    # print(X.shape)
    # print(y.shape)
    classes = []
    if y.dtype == object:
        # then we need to labelbinarize it
        le = LabelEncoder()
        y_notused = le.fit_transform(y)
        classes = le.classes_

    return X, y, classes


def train_model(model, X, y, name=None, param_grid=None):
    if name:
        print(f"Training: {name}")

    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X, y)
        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)
        _best_model = grid.best_estimator_
        _best_params = grid.best_params_
        _best_score = grid.best_score_
    else:
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(cv_scores, cv_scores.mean())
        _best_model = model
        _best_params = param_grid
        _best_score = cv_scores.mean()

    return _best_score, _best_params, _best_model


def create_logistic_regression_model():
    logreg = LogisticRegression(multi_class='multinomial')
    return logreg


def create_decision_tree():
    tree = DecisionTreeClassifier()
    return tree


def create_svc():
    svc = SVC(kernel='linear', C=1)
    return svc


def create_mnb():
    mnb = GaussianNB()
    return mnb


def create_knn():
    knn = KNeighborsClassifier()
    return knn


def create_linear():
    lin = LinearRegression()
    return lin


def find_best_model(X, y):
    models = [
        {
            'model': make_pipeline(StandardScaler(), create_logistic_regression_model()),
            'params_grid': dict(logisticregression__penalty=['l2'], logisticregression__C=[10, 1, 0.1, 0.01], logisticregression__solver=['newton-cg', 'sag', 'lbfgs'],
                                logisticregression__max_iter=[100, 200, 300]),
            'name': 'LogisticRegression',
            'skip': False
        },
        {
            'model': make_pipeline(StandardScaler(), create_decision_tree()),
            'params_grid': dict(decisiontreeclassifier__criterion=['gini', 'entropy'], decisiontreeclassifier__max_depth=[2, 3, 4, 5], decisiontreeclassifier__min_samples_split=[2, 3]),
            'name': 'DecisionTree',
            'skip': False
        },
        {
            'model': make_pipeline(StandardScaler(), create_svc()),
            'params_grid': dict(kernel=['linear', 'rbf', 'poly'], gamma=['auto', 'scale']),
            'name': 'SVC',
            'skip': True
        },
        {
            'model': make_pipeline(StandardScaler(), create_mnb()),
            'params_grid': None,
            'name': 'MultinomialNB',
            'skip': True
        },
        {
            'model': make_pipeline(StandardScaler(), create_knn()),
            'params_grid': dict(n_neighbors=list(range(1, 10)), weights=['uniform', 'distance']),
            'name': 'KNN GridSearch',
            'skip': True
        },
        {
            'model': make_pipeline(StandardScaler(), create_knn()),
            'params_grid': None,
            'name': 'KNN Default',
            'skip': True
        },
        {
            'model': make_pipeline(StandardScaler(), create_linear()),
            'params_grid': None,
            'name': 'Linear',
            'skip': True
        },
        {
            'model': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'params_grid': dict(randomforestclassifier__n_estimators=[100], randomforestclassifier__max_depth=[2,3,4]),
            'name': 'RandomForestClassifier',
            'skip': False
        },
        {
            'model': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
            'params_grid': None,
            'name': 'GradientBoostingClassifier',
            'skip': True

        },
        {
            'model': make_pipeline(StandardScaler(), MLPClassifier()),
            'params_grid': dict(activation=['relu'],
                                solver=['sgd', 'adam'],
                                alpha=[100, 10, 1], max_iter=[500, 600],
                                hidden_layer_sizes=[(X.shape[1], 128, 16), (X.shape[1], 100)]),
            'name': 'MLP',
            'skip': True
        }

    ]
    best_model = None
    best_params = None
    best_score = -1
    for model in models:
        if not model['skip']:
            score, params, best = train_model(model['model'], X, y, name=model['name'], param_grid=model['params_grid'])

            if score > best_score:
                best_params = params
                best_model = best
                best_score = score

    return best_model, best_params, best_score


"""
KNN Grid
0.72
{'n_neighbors': 23, 'weights': 'uniform'}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=23, p=2,
           weights='uniform')


MLP
0.7133333333333334
{'activation': 'relu', 'alpha': 10, 'hidden_layer_sizes': (250, 128, 16), 'solver': 'sgd'}
MLPClassifier(activation='relu', alpha=10, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(250, 128, 16), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='sgd', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
"""


def save_best_model(X, y):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=None, n_neighbors=23, p=2,
                               weights='uniform')
    knn.fit(X, y)

    joblib.dump(knn, f"{model_name}.pkl")

'''
python 02_pose_model_training.py --file-name ymca_training.csv --model-name ymca_pose_model

'''
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-data", type=str, required=False, default=str(1),
                    help="name of the training data file")
    ap.add_argument("--model-name", type=str, required=False, default=f'{model_name}',
                    help=f"name of the saved pickled model [no suffix]. Default: {model_name}.pkl")
    args = vars(ap.parse_args())
    
    
    mnum = args['training_data']
    model_name = f'./models/level{mnum}_pose_model'
    X, y, classes = get_data(f'./data/level{mnum}_training.csv')

    best_model, best_params, best_score = find_best_model(X, y)

    print("*******  Best Model and Parameters  *********")
    print(best_model)
    print(best_params)
    print(best_score)
    with open(f'{model_name}_metadata.txt', 'w') as f:
        f.write(f'{best_model}\n')
        f.write(f'{best_params}\n')
        f.write(f'{best_score}\n')

    with open(f'{model_name}_classes.txt', 'w') as f:
        f.write(f"{classes}")


    joblib.dump(best_model, f"{model_name}.pkl")

    print(f"Done saving model to best model:  {model_name}.pkl")
