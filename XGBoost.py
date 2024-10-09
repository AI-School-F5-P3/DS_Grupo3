import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_dataset4.csv')

X = df.drop('stroke', axis=1)
y = df['stroke'] 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


accuracies = []


for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
   
    model = xgb.XGBClassifier(
        n_estimators=178,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=2,
        subsample=0.93,
        colsample_bytree=0.72,
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)

   
    y_train_pred = model.predict(X_train_resampled)  
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    
    accuracies.append(test_accuracy)

   
    print(f"\nModel accuracy on training set: {train_accuracy * 100:.2f}%")
    print(f"Model accuracy on test set: {test_accuracy * 100:.2f}%")

    
    print("Classification report for the test set:")
    print(classification_report(y_test, y_test_pred))

    
    print("Confusion matrix for the test set:")
    print(confusion_matrix(y_test, y_test_pred))

    
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,  
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature importance:")
    print(importance_df)

    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

average_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy across folds: {average_accuracy:.4f}")

joblib.dump(scaler, 'model/scaler.joblib')

joblib.dump(model, 'model/xgboost_model.joblib')
