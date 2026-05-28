from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models():

    # Logistic Regression (baseline)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    # AdaBoost
    ada = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.3,
        random_state=42
    )

    # XGBoost (strong model)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )

    # LightGBM (best performer)
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=64,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )

    #ENSEMBLE
    ensemble = VotingClassifier(
    estimators=[
        ('lgb', lgb),
        ('xgb', xgb)
    ],
    voting='soft',
    weights=[3, 2]   # LightGBM stronger
)

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "AdaBoost": ada,
        "XGBoost": xgb,
        "LightGBM": lgb,
        "Ensemble Model": ensemble
    }

    return models