df_train = pd.read_csv('/content/drive/MyDrive/Sole/dataset/train.csv')
df_train.dropna(inplace=True)

y = df_train['Clade']
X = df_train.drop(['Clade', 'ID'], axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# توازن البيانات باستخدام SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
