from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Input:
        X            — feature matrix (all input columns)
        y            — target column (what we are predicting)
        test_size    — fraction of data kept for testing (default 20%)
        random_state — seed for reproducibility (same split every run)
    Output: X_train, X_test, y_train, y_test
    
    WHY SPLIT THE DATA?
        If you train and test on the same data, your model will appear accurate
        because it has already "seen" the answers. This is called overfitting.
        The test set simulates NEW, unseen customers — giving you an honest
        measure of how the model performs in the real world.
        
        80% training / 20% testing is the standard split.
        random_state=42 ensures everyone gets the same split (reproducibility).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train, y_train):
    """
    Train four regression models for claim severity prediction
    
    Input:  X_train, y_train — the training portion of the data
    Output: four trained model objects (lr, dt, rfr, xgb)
    
    HOW TRAINING WORKS:
        .fit(X_train, y_train) is the learning step.
        Each model looks at the features (X_train) and their known outcomes (y_train)
        and finds the mathematical relationship that best maps inputs → outputs.
        After .fit(), the model stores learned parameters internally and is
        ready to make predictions on new data using .predict().
    """
    
    # MODEL 1: Linear Regression (Parametric / Statistical)
    # Fits a straight line (or hyperplane in multiple dimensions):
    #   claims = β₀ + β₁·age + β₂·risk_index + ...
    # The model learns β coefficients that minimise the sum of squared errors.
    lr_model = LinearRegression()
    
    # MODEL 2: Decision Tree Regressor (Nonparametric)
    # Splits the data into branches using if/else rules:
    #   "If risk_index > 0.8 AND vehicle_age < 5 → predict high claims"
    # The tree grows by finding the split at each step that most reduces prediction error.
    dt_model = DecisionTreeRegressor(random_state=42)
    
    # MODEL 3: Random Forest Regressor (Ensemble / Nonparametric)
    # Builds many decision trees independently (default: 100 trees).
    # Each tree is trained on a random subset of rows AND a random subset of features.
    # Final prediction = AVERAGE of all trees' predictions.
    rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # MODEL 4: XGBoost Regressor (Gradient Boosting / Nonparametric)
    # Builds trees SEQUENTIALLY — each new tree learns from the ERRORS
    # (residuals) of all previous trees combined.
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    
    # TRAINING (the .fit() step for all four models)
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rfr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    return lr_model, dt_model, rfr_model, xgb_model


def train_classification_models(X_train, y_train):
    """
    Train four classification models for claim probability prediction
    
    Input:  X_train, y_train — the training portion of the data
    Output: four trained model objects (logr, dtc, rfc, xgbc)
    """
    
    # MODEL 1: Logistic Regression (Parametric)
    # Finds β coefficients for probability: P(claim) = 1/(1+e^-(β₀+β₁·age+...))
    logr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # MODEL 2: Decision Tree Classifier (Nonparametric)
    # Builds if/else rules for claim/no-claim decisions
    dtc_model = DecisionTreeClassifier(random_state=42)
    
    # MODEL 3: Random Forest Classifier (Ensemble)
    # Builds 100 trees and averages predictions
    rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # MODEL 4: XGBoost Classifier (Gradient Boosting)
    # Sequentially builds trees to correct previous errors
    xgbc_model = XGBClassifier(n_estimators=100, random_state=42)
    
    # TRAINING
    logr_model.fit(X_train, y_train)
    dtc_model.fit(X_train, y_train)
    rfc_model.fit(X_train, y_train)
    xgbc_model.fit(X_train, y_train)
    
    return logr_model, dtc_model, rfc_model, xgbc_model


def evaluate_severity_model(model, X_test, y_test):
    """
    Evaluate a regression model for claim severity prediction
    
    Input:
        model   — a trained model object
        X_test  — feature matrix from the test set (NOT seen during training)
        y_test  — true target values from the test set
    Output: mae, mse, r2 scores and the predicted values (y_pred)
    
    HOW EVALUATION WORKS:
        model.predict(X_test) → model generates predictions for each test row
        We compare those predictions against the actual known values (y_test)
        to calculate how accurate the model is on data it has NEVER seen before.
    """
    
    # Generate predictions on the held-out test set
    y_pred = model.predict(X_test)
    
    # METRIC 1: MAE — Mean Absolute Error
    # Formula: average of |y_actual - y_predicted| for every row
    # Units: same as the target variable (e.g., Rands)
    # Interpretation: "On average, our predictions are off by R X"
    mae = mean_absolute_error(y_test, y_pred)
    
    # METRIC 2: MSE — Mean Squared Error
    # Formula: average of (y_actual - y_predicted)² for every row
    # Units: squared units of the target (e.g., Rands²)
    # Interpretation: penalises large errors much more heavily than small ones
    mse = mean_squared_error(y_test, y_pred)
    
    # METRIC 3: R² — Coefficient of Determination
    # Formula: 1 - (sum of squared residuals / total sum of squares)
    # Range: 0 to 1 (can be negative for very bad models)
    # Interpretation: "The model explains X% of the variation in claims"
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, r2, y_pred


def evaluate_probability_model(model, X_test, y_test):
    """
    Evaluate a classification model for claim probability prediction
    
    Input:
        model   — a trained model object
        X_test  — feature matrix from the test set (NOT seen during training)
        y_test  — true target values from the test set
    Output: classification report, ROC-AUC score, predictions
    """
    
    # Generate predictions on the held-out test set
    y_pred = model.predict(X_test)
    
    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred_proba = None
        auc = 0.5
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return report, auc, y_pred, y_pred_proba


def plot_metrics(models, mae_scores, mse_scores, r2_scores):
    """
    Plot model comparison bar charts
    
    Input:
        models     — list of model name strings (for x-axis labels)
        mae_scores — list of MAE values, one per model
        mse_scores — list of MSE values, one per model
        r2_scores  — list of R² values, one per model
    Output: three bar charts displayed side by side
    
    WHY VISUALISE:
        A table of numbers is harder to read at a glance than a bar chart.
        Visualising metrics makes it immediately obvious which model wins
        on each metric — especially useful for presentations.
    """
    
    # CHART 1: MAE Comparison
    # Lower bar = better (fewer prediction errors on average)
    plt.figure(figsize=(6, 4))
    plt.bar(models, mae_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of MAE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # CHART 2: MSE Comparison
    # Lower bar = better
    # MSE values are typically much larger than MAE (squared units)
    plt.figure(figsize=(6, 4))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # CHART 3: R² Comparison
    # Higher bar = better (explains more variance)
    # Maximum possible value = 1.0 (perfect model)
    plt.figure(figsize=(6, 4))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.title('Comparison of R² Scores — Higher is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models
    
    Input:
        model        — a trained model object
        feature_names — list of feature names
        model_name   — name of the model for the title
    """
    try:
        # Try to get feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame(
                model.feature_importances_, 
                index=feature_names, 
                columns=["Importance"]
            )
        else:
            # Fallback for models without feature_importances_
            feature_importance = pd.DataFrame(
                [0.1] * len(feature_names), 
                index=feature_names, 
                columns=["Importance"]
            )
    except:
        feature_importance = pd.DataFrame(
            [0.1] * len(feature_names), 
            index=feature_names, 
            columns=["Importance"]
        )
    
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importance.head(10).plot(kind='barh', legend=False, color='skyblue')
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.show()


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract and return top N important features for a model
    
    Input:
        model          — trained tree-based model (Random Forest, XGBoost, etc.)
        feature_names  — list of feature names
        top_n          — number of top features to return (default: 10)
    
    Output:
        DataFrame with Feature and Importance columns, sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return feature_importance
    else:
        raise ValueError("Model does not have feature_importances_ attribute")


def calculate_risk_based_premium(claim_probability, predicted_severity, 
                                 expense_loading=100, profit_margin_rate=0.20,
                                 baseline_claim_rate=None):
    """
    Calculate risk-based premium using predicted probability and severity
    
    Input:
        claim_probability      — predicted P(claim) for a policy
        predicted_severity     — predicted claim amount IF claim occurs
        expense_loading        — fixed expense per policy
        profit_margin_rate     — profit margin as fraction of expected claims
        baseline_claim_rate    — overall claim rate for normalization
    
    Output:
        risk_adjusted_premium  — calculated premium for the policy
    
    Formula:
        Premium = (P(claim) × E[Severity]) + Expenses + Profit Margin
    """
    expected_claims = claim_probability * predicted_severity
    profit_component = expected_claims * profit_margin_rate
    premium = expected_claims + expense_loading + profit_component
    
    return premium


def calculate_pricing_metrics(y_true, y_pred_prob, y_pred_severity_avg, 
                             expense_loading=100, profit_margin_rate=0.20):
    """
    Calculate comprehensive pricing metrics
    
    Input:
        y_true              — actual claim indicators (0/1)
        y_pred_prob         — predicted probabilities of claim
        y_pred_severity_avg — average predicted severity for claims
        expense_loading     — fixed expense per policy
        profit_margin_rate  — profit margin rate
    
    Output:
        Dictionary with pricing metrics and statistics
    """
    # Calculate baseline metrics
    actual_claim_rate = y_true.mean()
    
    # Calculate premiums
    premiums = []
    for prob in y_pred_prob:
        premium = calculate_risk_based_premium(
            prob, y_pred_severity_avg, 
            expense_loading, profit_margin_rate,
            actual_claim_rate
        )
        premiums.append(premium)
    
    premiums = np.array(premiums)
    
    return {
        'mean_premium': premiums.mean(),
        'min_premium': premiums.min(),
        'max_premium': premiums.max(),
        'std_premium': premiums.std(),
        'premium_range': premiums.max() - premiums.min(),
        'risk_differentiation': premiums.max() / premiums.min(),
        'actual_claim_rate': actual_claim_rate
    }


def compare_models_summary(models_dict, X_test, y_test, model_type='regression'):
    """
    Create a summary comparison of multiple models
    
    Input:
        models_dict   — dictionary of {model_name: trained_model}
        X_test        — test feature matrix
        y_test        — test target values
        model_type    — 'regression' or 'classification'
    
    Output:
        DataFrame with comparison metrics for all models
    """
    results = []
    
    for model_name, model in models_dict.items():
        if model_type == 'regression':
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'R² Score': r2
            })
        
        elif model_type == 'classification':
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = 0.5
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': auc
            })
    
    return pd.DataFrame(results)


def create_risk_segments(probabilities, n_segments=10):
    """
    Create risk segments (deciles) from predicted probabilities
    
    Input:
        probabilities  — predicted claim probabilities
        n_segments     — number of risk segments (default: 10 for deciles)
    
    Output:
        Series with segment assignments for each observation
    """
    segments = pd.qcut(probabilities, q=n_segments, duplicates='drop', labels=False) + 1
    return segments


def analyze_segment_performance(y_true, y_pred_prob, predictions_claim, n_segments=10):
    """
    Analyze model performance by risk segment
    
    Input:
        y_true           — actual claim indicators
        y_pred_prob      — predicted probabilities
        predictions_claim — predicted claim indicators
        n_segments       — number of segments
    
    Output:
        DataFrame with performance metrics by segment
    """
    segments = create_risk_segments(y_pred_prob, n_segments)
    
    segment_analysis = []
    for seg in sorted(segments.unique()):
        mask = segments == seg
        
        segment_analysis.append({
            'Segment': seg,
            'Actual_Claims': y_true[mask].sum(),
            'Predicted_Claims': predictions_claim[mask].sum(),
            'Segment_Size': mask.sum(),
            'Actual_Rate': y_true[mask].mean(),
            'Predicted_Rate': y_pred_prob[mask].mean()
        })
    
    return pd.DataFrame(segment_analysis)