import pandas as pd
import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# загрузка модели и теста
model = joblib.load('dashboard/model/model.pkl')
X_test = pd.read_csv('dashboard/data/X_test.csv', index_col=0)
y_test = pd.read_csv('dashboard/data/y_test.csv', index_col=0).iloc[:, 0]

# создаём explainer
explainer = ClassifierExplainer(model, X_test, y_test, labels=['No purchase', 'Purchase'], model_output='probability')

# создаём dashboard
db = ExplainerDashboard(explainer, shap_interaction=False)

# сохраняем конфиг и explainer
db.to_yaml(
    "/app/dashboard/dashboard.yaml",
    explainerfile="/app/dashboard/explainer.joblib",
    dump_explainer=True
)
