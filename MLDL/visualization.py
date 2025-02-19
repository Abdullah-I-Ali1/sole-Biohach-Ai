results_df = pd.DataFrame(eval_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"])
results_df.to_csv("model_performance.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df["Model"], y=results_df["Accuracy"], palette="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(X, columns=df_train.columns[:-2])[feature_importances.index])
plt.title("Box Plot for Top Features")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X, columns=df_train.columns[:-2]).corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_test, rf_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(y_new_preds_clades, palette="viridis")
plt.title(f"Predicted Clade Distribution ({model_name})")
plt.xticks(rotation=45)
plt.show()
