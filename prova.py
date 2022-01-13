import joblib


model = joblib.load('./Monitoring/Complement_incremental_model_peak_6.pkl')

print(model.best_estimator_[3].n_features_in_)
