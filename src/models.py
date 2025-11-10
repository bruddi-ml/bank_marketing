import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import joblib 

class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()
        self.feature_set = {}
        self.target_col = 'target'  

    def add_features(self, flag=1):
        
        df = self.df.copy()

        if flag == 0:
            self.df = df
            return self.df

        #Feature indicating whether the customer was previously contacted
        df["was_previously_contacted"] = (df["N_last_days"] != 999).astype(int)

        #Bucketizing N_last_days into categorical feature
        df["contact_gap_bucket"] = pd.cut(df["N_last_days"], bins=[-1, 5, 10, 20, 40, 999], labels=["<5d", "5_10d", "10_20d", "20_40d", "never"])

        #Feature indicating whether a customer was contacted multiple times before
        df["contacted_multiple_times"] = (df["nb_previous_contact"] > 1).astype(int)

        #Feature indicating whether the contact was made on weekend
        df["is_weekend_call"] = df["week_day"].isin(["saturday", "sunday"]).astype(int)

        #Feature counting the number of loans a customer has
        df["loan_count"] = df[["has_credit", "housing_loan", "personal_loan"]].apply(lambda row: sum(val == "yes" for val in row), axis=1)

        #Bucketizing age into categorical feature
        df["age_bucket"] = pd.cut(df["age"], bins=[-1, 24, 34, 44, 54, 64, float('inf')], labels=['under_25', '25_34', '35_44', '45_54', '55_64', '65_plus'])

        self.df = df
        return self.df

    def generate_feature_sets(self):
        """
        Define multiple versions of numerical + categorical feature sets.
        """
        ### V1: Original features
        num_features_1 = ['age', 'last_contact_duration', 'contacts_per_campaign','N_last_days', 'nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees']
    
        ### V2: Removing last_contact_duration data leakage
        num_features_2 = ['age', 'contacts_per_campaign','N_last_days', 'nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees']

        ### V3: Removing contacts_per_campaign: test model against V2
        num_features_3 = ['age','N_last_days', 'nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees']

        ### V4: Adding feature was_previously_contacted
        num_features_4 = ['age', 'contacts_per_campaign','was_previously_contacted','N_last_days', 'nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees']

        ### V5: Removing N_last_days, using was_previously_contacted instead
        num_features_5 = ['age', 'contacts_per_campaign','was_previously_contacted','nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees']

        ### V6: Adding feature contacted_multiple_times
        num_features_6 = ['age', 'contacts_per_campaign','was_previously_contacted','nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees', 'contacted_multiple_times']

        ### V7: Adding feature is_weekend_call
        num_features_7 = ['age', 'contacts_per_campaign','was_previously_contacted','nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees', 'contacted_multiple_times', 'is_weekend_call']

        ### V8: Adding feature loan_count
        num_features_8 = ['age', 'contacts_per_campaign','was_previously_contacted','nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']

        ### V9: Replacing age with age_bucket categorical feature
        num_features_9 = ['was_previously_contacted','contacts_per_campaign','nb_previous_contact','emp_var_rate', 'cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_employees', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']
    
        ### V10: nb_employees, euri_3_month and emp_var_rate are highly correlated. Rotating which one to remove.
        ### V10.1: Keeping nb_employees
        num_features_10_1 = ['contacts_per_campaign','was_previously_contacted','nb_previous_contact','cons_price_index', 'cons_conf_index', 'nb_employees', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']

        ### V10.2: Keeping emp_var_rate
        num_features_10_2 = ['contacts_per_campaign','was_previously_contacted','nb_previous_contact','cons_price_index', 'cons_conf_index', 'emp_var_rate', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']

        ### V10.3:  Keeping euri_3_month 
        num_features_10_3 = ['contacts_per_campaign','was_previously_contacted','nb_previous_contact','cons_price_index', 'cons_conf_index', 'euri_3_month', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']

        ### V11: nb_previous_contact is correlated with contacted_multiple_times. Rotating which one to remove.
        ### V11.1: Keeping contacted_multiple_times
        num_features_11_1 = ['contacts_per_campaign','was_previously_contacted','cons_price_index', 'cons_conf_index', 'euri_3_month', 'contacted_multiple_times', 'is_weekend_call', 'loan_count']

        ### V11.2: Keeping nb_previous_contact
        num_features_11_2 = ['contacts_per_campaign','was_previously_contacted','cons_price_index', 'cons_conf_index', 'euri_3_month', 'nb_previous_contact', 'is_weekend_call', 'loan_count']

        ### V12: cons_price_index and euri_3_month are correlated. Rotating which one to remove (if at all).
        ### V12.1: Removing cons_price_index
        num_features_12_1 = ['contacts_per_campaign','was_previously_contacted', 'cons_conf_index', 'euri_3_month', 'nb_previous_contact', 'is_weekend_call', 'loan_count']

        ### V12.2: Removing euri_3_month
        num_features_12_2 = ['contacts_per_campaign','was_previously_contacted', 'cons_conf_index', 'cons_price_index', 'nb_previous_contact', 'is_weekend_call', 'loan_count']
    

        ### V1: All categorical features from the original dataset
        cat_features_1 = ['occupation', 'marital_status', 'education','has_credit', 'housing_loan', 'personal_loan','contact_mode', 'month', 'week_day', 'previous_outcome']

        ### V5: Replacing numerical N_last_days with categorical contact_gap_bucket
        cat_features_2 = ['occupation', 'marital_status', 'education','has_credit', 'housing_loan', 'personal_loan','contact_mode', 'month', 'week_day', 'previous_outcome', 'contact_gap_bucket']

        ### V8: Removing has_credit, housing_loan, personal_loan, instead using loan_count numerical feature
        cat_features_3 = ['occupation', 'marital_status', 'education','contact_mode', 'month', 'week_day', 'previous_outcome', 'contact_gap_bucket']

        ### V9: Adding age_bucket categorical feature, replacing numerical age
        cat_features_4 = ['occupation', 'marital_status', 'education','contact_mode', 'month', 'week_day', 'previous_outcome', 'contact_gap_bucket', 'age_bucket']

        self.feature_sets = {
            "v1": (num_features_1, cat_features_1),             # baseline
            "v2": (num_features_2, cat_features_1),             # no leakage: remove last_contact_duration
            "v3": (num_features_3, cat_features_1),             # explore potential leakage feature: remove contacts_per_campaign
            "v4": (num_features_4, cat_features_1),             # add new num. feature: was_previously contacted
            "v5.0": (num_features_5, cat_features_1),           # remove N_last_days, using was_previously_contacted instead
            "v5.1": (num_features_5, cat_features_2),           # replace N_last_days with categorical contact_gap_bucket
            "v6": (num_features_6, cat_features_2),             # add feature: contacted_multiple_times
            "v7": (num_features_7, cat_features_2),             # add feature: is_weekend_call
            "v8.0": (num_features_8, cat_features_2),           # add feature: loan_count
            "v8.1": (num_features_8, cat_features_3),           # replace individual loan types, keep loan_count
            "v9": (num_features_9, cat_features_4),             # replace age with age_bucket
            "10.0": (num_features_10_1, cat_features_4),        # (1/3) Rotate among correlated values. Keep nb_employees
            "10.1": (num_features_10_2, cat_features_4),        # (2/3) Rotate among correlated values. Keep emp_var_rate  
            "10.2": (num_features_10_3, cat_features_4),        # (3/3) Rotate among correlated values. Keep euri_3_month 
            "11.0": (num_features_11_1, cat_features_4),        # (1/2) Rotate among correlated values. Keep contacted_multiple_times
            "11.1": (num_features_11_2, cat_features_4),        # (2/2) Rotate among correlated values. Keep nb_previous_contact
            "12.0": (num_features_12_1, cat_features_4),        # (1/2) Rotate among correlated values. Keep euri_3_month
            "12.1": (num_features_12_2, cat_features_4),        # (2/2) Rotate among correlated values. Keep cons_price_index
        }

        return self.feature_sets

    def correlation_matrix(self):
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlation Matrix of Numerical Features")
        plt.tight_layout()
        plt.show()

class LogisticRegressionModel:

    def __init__(self, df, num_features, cat_features, target_col):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_col = target_col
        self.model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = df
        

    def build_pipeline(self):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.num_features),
            ("cat", categorical_transformer, self.cat_features)
        ])

        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        return self.model

    def train(self):
        self.X = self.df[self.num_features + self.cat_features]
        self.y = self.df[self.target_col].apply(lambda val: 1 if val == "yes" else 0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        if self.model is None:
            self.build_pipeline()

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, "trained_models\\model_logreg.pkl")
        print("Model trained and saved as 'model_logreg.pkl'")

    def evaluate(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n--- Evaluation Report ---")
        print(classification_report(self.y_test, y_pred))
        print("ROC AUC:", roc_auc_score(self.y_test, y_prob))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print("F1:", f1_score(self.y_test, y_pred))
        print("PR AUC:", average_precision_score(self.y_test, y_prob))
        print("Log Loss:", log_loss(self.y_test, y_prob))
        print("Brier Score:", brier_score_loss(self.y_test, y_prob))

    def cross_validate(self, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {"auc": [], "precision": [], "recall": [], "f1": []}

        print("\n--- Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model_fold = self.build_pipeline()
            model_fold.fit(X_train, y_train)
            y_prob = model_fold.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, y_prob))
            metrics["precision"].append(precision_score(y_val, y_pred))
            metrics["recall"].append(recall_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred))

            print(f"Fold {fold+1}: AUC={metrics['auc'][-1]:.4f}, "
                  f"Precision={metrics['precision'][-1]:.4f}, "
                  f"Recall={metrics['recall'][-1]:.4f}, "
                  f"F1={metrics['f1'][-1]:.4f}")

        print("\n--- Final Cross-Validation Results ---")
        for key, values in metrics.items():
            print(f"Mean {key.upper()}: {np.mean(values):.4f}")

    def plot_confusion_matrix(self):

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
        plt.show()

    def plot_important_features(self):

        importance = self.model.named_steps["classifier"].coef_[0]
        feature_names = self.preprocessor.get_feature_names_out()
        imp_df = pd.DataFrame({"feature": feature_names, "coef": importance})
        imp_df = imp_df.sort_values("coef", key=abs, ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="coef", y="feature", data=imp_df.head(15))
        plt.title("Top 15 Feature Coefficients")
        plt.tight_layout()
        plt.show()

    def test_thresholds(self):

        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        for t in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_prob >= t).astype(int)
            print(f"\nThreshold: {t:.1f}")
            print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
            print(f"Recall:    {recall_score(self.y_test, y_pred):.3f}")
            print(f"F1 Score:  {f1_score(self.y_test, y_pred):.3f}")

class LightGBMModel:

    def __init__(self, df, num_features, cat_features, target_col):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_col = target_col
        self.model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = df.copy()

    def build_pipeline(self):

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.num_features),
            ("cat", categorical_transformer, self.cat_features)
        ])

        self.model = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", LGBMClassifier(random_state=42,class_weight='balanced',n_estimators=300,learning_rate=0.05,max_depth=6, verbose=-1))
        ])

        return self.model

    def train(self):

        features = self.num_features + self.cat_features
        self.df[self.cat_features] = self.df[self.cat_features].astype("category")
        self.X = self.df[features]
        self.y = self.df[self.target_col].apply(lambda x: 1 if x == 'yes' else 0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        if self.model is None:
            self.build_pipeline()

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, "trained_models\\model_lgbm.pkl") ### needed for the API application
        print("Model trained and saved as 'mmodel_lgbm.pkl'")

    def evaluate(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n--- Evaluation Report ---")
        print(classification_report(self.y_test, y_pred))
        print("ROC AUC:", roc_auc_score(self.y_test, y_prob))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print("F1:", f1_score(self.y_test, y_pred))
        print("PR AUC:", average_precision_score(self.y_test, y_prob))
        print("Log Loss:", log_loss(self.y_test, y_prob))
        print("Brier Score:", brier_score_loss(self.y_test, y_prob))

    def cross_validate(self, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {"auc": [], "precision": [], "recall": [], "f1": []}

        print("\n--- Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model_fold = self.build_pipeline()
            model_fold.fit(X_train, y_train)
            y_prob = model_fold.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, y_prob))
            metrics["precision"].append(precision_score(y_val, y_pred))
            metrics["recall"].append(recall_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred))

            print(f"Fold {fold+1}: AUC={metrics['auc'][-1]:.4f}, "
                  f"Precision={metrics['precision'][-1]:.4f}, "
                  f"Recall={metrics['recall'][-1]:.4f}, "
                  f"F1={metrics['f1'][-1]:.4f}")

        print("\n--- Final Cross-Validation Results ---")
        for key, values in metrics.items():
            print(f"Mean {key.upper()}: {np.mean(values):.4f}")

    def test_thresholds(self):

        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        for t in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_prob >= t).astype(int)
            print(f"\nThreshold: {t:.1f}")
            print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
            print(f"Recall:    {recall_score(self.y_test, y_pred):.3f}")
            print(f"F1 Score:  {f1_score(self.y_test, y_pred):.3f}")

    def plot_feature_importance(self):
        fitted_model = self.model.named_steps["classifier"]
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        importances = fitted_model.feature_importances_

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df.head(20), x="importance", y="feature")
        plt.title("Top 20 LightGBM Feature Importances")
        plt.tight_layout()
        plt.show()

class RandomForestModel:

    def __init__(self, df, num_features, cat_features, target_col):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_col = target_col
        self.model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = df

    def build_pipeline(self):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.num_features),
            ("cat", categorical_transformer, self.cat_features)
        ])

        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                class_weight="balanced",
                random_state=42
            ))
        ])

        return self.model

    def train(self):
        self.X = self.df[self.num_features + self.cat_features]
        self.y = self.df[self.target_col].apply(lambda val: 1 if val == "yes" else 0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        if self.model is None:
            self.build_pipeline()

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, "trained_models\\model_rand_forest.pkl")
        print("Model trained and saved as 'model_rand_forest.pkl'")

    def evaluate(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n--- Evaluation Report ---")
        print(classification_report(self.y_test, y_pred))
        print("ROC AUC:", roc_auc_score(self.y_test, y_prob))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print("F1:", f1_score(self.y_test, y_pred))
        print("PR AUC:", average_precision_score(self.y_test, y_prob))
        print("Log Loss:", log_loss(self.y_test, y_prob))
        print("Brier Score:", brier_score_loss(self.y_test, y_prob))

    def cross_validate(self, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {"auc": [], "precision": [], "recall": [], "f1": []}

        print("\n--- Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model_fold = self.build_pipeline()
            model_fold.fit(X_train, y_train)
            y_prob = model_fold.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, y_prob))
            metrics["precision"].append(precision_score(y_val, y_pred))
            metrics["recall"].append(recall_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred))

            print(f"Fold {fold+1}: AUC={metrics['auc'][-1]:.4f}, "
                  f"Precision={metrics['precision'][-1]:.4f}, "
                  f"Recall={metrics['recall'][-1]:.4f}, "
                  f"F1={metrics['f1'][-1]:.4f}")

        print("\n--- Final Cross-Validation Results ---")
        for key, values in metrics.items():
            print(f"Mean {key.upper()}: {np.mean(values):.4f}")

    def test_thresholds(self):

        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        for t in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_prob >= t).astype(int)
            print(f"\nThreshold: {t:.1f}")
            print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
            print(f"Recall:    {recall_score(self.y_test, y_pred):.3f}")
            print(f"F1 Score:  {f1_score(self.y_test, y_pred):.3f}")

    def plot_feature_importance(self):
        rf_model = self.model.named_steps['classifier']

        ohe = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = ohe.get_feature_names_out(self.cat_features)
        all_feature_names = np.concatenate([self.num_features, cat_feature_names])

        feature_importances = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importances.head(20), x='Importance', y='Feature')
        plt.title('Top 20 Feature Importances - Random Forest')
        plt.tight_layout()
        plt.show()

class VotingClassifierModel:

    def __init__(self, df, num_features, cat_features, target_col):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_col = target_col
        self.model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = df

    def build_pipeline(self):

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine them
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.num_features),
            ('cat', categorical_transformer, self.cat_features)
        ])

        logreg = LogisticRegression(max_iter=1000, class_weight='balanced',random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
        lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', VotingClassifier(
                estimators=[
                    ('lr', logreg),
                    ('rf', rf),
                    ('lgbm', lgbm)
                ],
                voting='soft'  # Use predicted probabilities (better for imbalance)
            ))
        ])

        return self.model

    def train(self):
        self.X = self.df[self.num_features + self.cat_features]
        self.y = self.df[self.target_col].apply(lambda val: 1 if val == "yes" else 0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        if self.model is None:
            self.build_pipeline()

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, "trained_models\\model_voting_classifier.pkl")
        print("Model trained and saved as 'model_voting_classifier.pkl'")

    def evaluate(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n--- Evaluation Report ---")
        print(classification_report(self.y_test, y_pred))
        print("ROC AUC:", roc_auc_score(self.y_test, y_prob))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print("F1:", f1_score(self.y_test, y_pred))
        print("PR AUC:", average_precision_score(self.y_test, y_prob))
        print("Log Loss:", log_loss(self.y_test, y_prob))
        print("Brier Score:", brier_score_loss(self.y_test, y_prob))

    def cross_validate(self, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {"auc": [], "precision": [], "recall": [], "f1": []}

        print("\n--- Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model_fold = self.build_pipeline()
            model_fold.fit(X_train, y_train)
            y_prob = model_fold.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, y_prob))
            metrics["precision"].append(precision_score(y_val, y_pred))
            metrics["recall"].append(recall_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred))

            print(f"Fold {fold+1}: AUC={metrics['auc'][-1]:.4f}, "
                  f"Precision={metrics['precision'][-1]:.4f}, "
                  f"Recall={metrics['recall'][-1]:.4f}, "
                  f"F1={metrics['f1'][-1]:.4f}")

        print("\n--- Final Cross-Validation Results ---")
        for key, values in metrics.items():
            print(f"Mean {key.upper()}: {np.mean(values):.4f}")

    def test_thresholds(self):

        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        for t in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_prob >= t).astype(int)
            print(f"\nThreshold: {t:.1f}")
            print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
            print(f"Recall:    {recall_score(self.y_test, y_pred):.3f}")
            print(f"F1 Score:  {f1_score(self.y_test, y_pred):.3f}")

    def individual_model_performance(self):
        voting_clf = self.model.named_steps['classifier']

        model_names = voting_clf.named_estimators_.keys()
        model_list = voting_clf.named_estimators_.values()

        X_test_transformed = self.model.named_steps['preprocessor'].transform(self.X_test)

        for name, model in zip(model_names, model_list):
            y_pred_individual = model.predict(X_test_transformed)
            acc = roc_auc_score(self.y_test, y_pred_individual)
            print(f"{name} roc auc: {acc:.4f}")

class FeatureEvaluator:

    def __init__(self, df, fe, model_class, target_metric='roc_auc'): 
        """
        df: pandas DataFrame
        feature_engineer: instance of FeatureEngineer
        model_class: model class to instantiate (e.g., LogisticRegressionModel)
        target_metric: which metric to optimize ("roc_auc", "precision", "recall", "f1", "pr_auc")
        """
        self.df = df.copy()
        self.fe = fe
        self.model_class = model_class
        self.target_metric = target_metric
        self.results = []

    def evaluate_all(self):
        feature_sets = self.fe.generate_feature_sets()

        for version, (num_features, cat_features) in feature_sets.items():
            print(f"\nEvaluating Feature Set Version: {version}")
            if version == "v1":
                print("\n Skipping baseline version v1.")
                continue

            model = self.model_class(self.df, num_features, cat_features, target_col='target')
            model.train()

            y_prob = model.model.predict_proba(model.X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            roc_auc = roc_auc_score(model.y_test, y_prob)
            precision = precision_score(model.y_test, y_pred)
            recall = recall_score(model.y_test, y_pred)
            f1 = f1_score(model.y_test, y_pred)
            pr_auc = average_precision_score(model.y_test, y_prob)

            self.results.append({
                "version": version,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_auc": pr_auc
            })

        results_df = pd.DataFrame(self.results)
        print("\n--- Feature Set Evaluation Results ---")
        print(results_df.sort_values(by=self.target_metric, ascending=False))

def main():

    # Load dataset
    df = pd.read_csv('data\\bank_dataset.csv')

    # FeaTure Engineering
    fe = FeatureEngineer(df)
    df = fe.add_features(flag=1)
    feature_sets = fe.generate_feature_sets()
    num_features, cat_features = feature_sets["v9"]
    

    # logistic regression model
    
    logreg_model = LogisticRegressionModel(df, num_features, cat_features, target_col='target')
    logreg_model.train()
    logreg_model.evaluate()
    logreg_model.cross_validate(k=5)
    logreg_model.plot_confusion_matrix()
    logreg_model.plot_important_features()
    logreg_model.test_thresholds()
    

    # LightGBM model
    
    lgbm_model = LightGBMModel(df, num_features, cat_features, target_col='target')
    lgbm_model.train()
    lgbm_model.evaluate()
    lgbm_model.cross_validate(k=5)
    lgbm_model.test_thresholds()
    lgbm_model.plot_feature_importance()
    

    # Random Forest model
    
    rf_model = RandomForestModel(df, num_features, cat_features, target_col='target')
    rf_model.train()
    rf_model.evaluate()
    rf_model.cross_validate(k=5)
    rf_model.test_thresholds()
    rf_model.plot_feature_importance()
    

    # Voting Classifier model
    
    voting_model = VotingClassifierModel(df, num_features, cat_features, target_col='target')
    voting_model.train()
    voting_model.evaluate()
    voting_model.cross_validate(k=5)
    voting_model.test_thresholds()
    voting_model.individual_model_performance()
    

    # Feature Set Evaluation: Evaluating all feature sets with different models
    # a. Logistic Regression
    feature_evaluator = FeatureEvaluator(df, fe, model_class=LogisticRegressionModel, target_metric='precision')
    feature_evaluator.evaluate_all()
    # b. LightGBM
    feature_evaluator = FeatureEvaluator(df, fe, model_class=LightGBMModel, target_metric='roc_auc')
    feature_evaluator.evaluate_all()
    # c. Random Forest
    feature_evaluator = FeatureEvaluator(df, fe, model_class=RandomForestModel, target_metric='roc_auc')
    feature_evaluator.evaluate_all()
    # d. Voting Classifier
    feature_evaluator = FeatureEvaluator(df, fe, model_class=VotingClassifierModel, target_metric='roc_auc')
    feature_evaluator.evaluate_all()




