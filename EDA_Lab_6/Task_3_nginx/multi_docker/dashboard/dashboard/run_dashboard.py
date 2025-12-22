from explainerdashboard import ExplainerDashboard

if __name__=='__main__':    
    db = ExplainerDashboard.from_config('dashboard/explainer.joblib', "dashboard/dashboard.yaml")
    db.run(host='0.0.0.0', port=9050, use_waitress=True)