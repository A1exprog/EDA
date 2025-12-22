# model.py
import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

def train_model_full_pipeline(X, y, model, param_grid, test_size, save_path, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_test.to_csv('/home/alexus/Desktop/DeepCA/EDA-main/data/X_test.csv')
    y_test.to_csv('/home/alexus/Desktop/DeepCA/EDA-main/data/y_test.csv')

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=8)
    grid_search.fit(X_train, y_train)

    print(f"Лучшие параметры GridSearch: {grid_search.best_params_}")
    print(f"Лучшая метрика ROC-AUC для GridSearch на CV: {grid_search.best_score_}")
    
    y_pred = grid_search.best_estimator_.predict(X_test)
    print(f"Итоговая метрика ROC-AUC на тесте: {roc_auc_score(y_test, y_pred)}")
    joblib.dump(grid_search.best_estimator_, save_path)

if __name__=='__main__':
    df = pd.read_csv('/home/alexus/Desktop/DeepCA/EDA-main/data/df_for_model.csv', index_col=0)
    df = pd.concat([
        df[df['Revenue'] == False].sample(2500), 
        df[df['Revenue'] == True]
    ])
    
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(df.drop(columns=['Revenue']), df['Revenue'])
    params_for_gradient_boosting = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5],
        'n_estimators': [75, 100, 150]
    }

    train_model_full_pipeline(X, y, GradientBoostingClassifier(random_state=42), params_for_gradient_boosting, 0.2, save_path='/home/alexus/Desktop/DeepCA/EDA-main/dashboard/model.pkl')