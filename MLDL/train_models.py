def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average='weighted', multi_class='ovr')
    print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    return [model_name, acc, prec, rec, f1, auc]

eval_results = []

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
eval_results.append(evaluate_model("Random Forest", y_test, rf_preds))

# LightGBM
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
eval_results.append(evaluate_model("LightGBM", y_test, lgbm_preds))

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
eval_results.append(evaluate_model("XGBoost", y_test, xgb_preds))

# Neural Network
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

adamw_optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)
nn_model.compile(optimizer=adamw_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1)
y_nn_preds = np.argmax(nn_model.predict(X_test), axis=1)
eval_results.append(evaluate_model("Neural Network", y_test, y_nn_preds))
