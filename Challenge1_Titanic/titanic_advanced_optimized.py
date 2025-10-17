import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


# BƯỚC 1: TẢI VÀ GỘP DỮ LIỆU
print("🚢 Bắt đầu xử lý dữ liệu Titanic (Phiên bản Advanced Tối Ưu)...")

# Tải dữ liệu
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Lưu lại các thông tin cần thiết
train_labels = train_df['Survived']
test_ids = test_df['PassengerId']

# Gộp train và test để xử lý đồng nhất
full_df = pd.concat([
    train_df.drop(columns=['Survived', 'PassengerId']),
    test_df.drop(columns=['PassengerId'])
], ignore_index=True)

print("✅ Tải và gộp dữ liệu thành công.")

# BƯỚC 2: FEATURE ENGINEERING SIÊU NÂNG CAO

# --- 2.1. Xử lý giá trị thiếu cơ bản ---
most_frequent_port = full_df['Embarked'].mode()[0]
full_df['Embarked'] = full_df['Embarked'].fillna(most_frequent_port)
full_df['Fare'] = full_df['Fare'].fillna(full_df['Fare'].median())

# --- 2.2. Tạo đặc trưng 'Deck' từ 'Cabin' ---
full_df['Deck'] = full_df['Cabin'].str[0].fillna('U')

# --- 2.3. Tạo đặc trưng 'Title' từ 'Name' (cải tiến) ---
full_df['Title'] = full_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
    'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 
    'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
}
full_df['Title'] = full_df['Title'].replace(title_mapping)

# --- 2.4. Tạo các đặc trưng Family ---
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
full_df['IsAlone'] = (full_df['FamilySize'] == 1).astype(int)
full_df['SmallFamily'] = ((full_df['FamilySize'] >= 2) & (full_df['FamilySize'] <= 4)).astype(int)
full_df['LargeFamily'] = (full_df['FamilySize'] > 4).astype(int)

# --- 2.5. Tạo đặc trưng Ticket ---
ticket_counts = full_df['Ticket'].value_counts()
full_df['TicketFreq'] = full_df['Ticket'].map(ticket_counts)
full_df['TicketGroup'] = full_df['TicketFreq'].apply(lambda x: 'Single' if x == 1 else 'Group')

# --- 2.6. Tạo đặc trưng Age (nâng cao) ---
def get_age_group(age):
    if pd.isna(age): return 'Unknown'
    elif age < 1: return 'Infant'
    elif age < 5: return 'Toddler'
    elif age < 12: return 'Child'
    elif age < 18: return 'Teen'
    elif age < 30: return 'Young_Adult'
    elif age < 50: return 'Adult'
    elif age < 65: return 'Middle_Age'
    else: return 'Senior'

# --- 2.7. Tạo đặc trưng Fare (nâng cao) ---
def get_fare_group(fare):
    if fare <= 7.91: return 'Low'
    elif fare <= 14.45: return 'Medium'
    elif fare <= 31: return 'High'
    else: return 'Very_High'

# Tạo Fare per person
full_df['FarePerPerson'] = full_df['Fare'] / full_df['FamilySize']

# --- 2.8. Tạo các đặc trưng tương tác quan trọng ---
full_df['Pclass_Sex'] = full_df['Pclass'].astype(str) + '_' + full_df['Sex']
full_df['Pclass_Age'] = full_df['Pclass'].astype(str) + '_' + full_df['Age'].apply(lambda x: 'Child' if x < 16 else 'Adult')
full_df['Sex_Age'] = full_df['Sex'] + '_' + full_df['Age'].apply(lambda x: 'Child' if x < 16 else 'Adult')

# --- 2.9. Tạo đặc trưng Cabin ---
full_df['HasCabin'] = full_df['Cabin'].notna().astype(int)
full_df['CabinNumber'] = full_df['Cabin'].str.extract(r'(\d+)').astype(float)
full_df['CabinNumber'] = full_df['CabinNumber'].fillna(0)

# --- 2.10. Tạo đặc trưng từ Ticket (nâng cao) ---
# Lấy số từ ticket
full_df['TicketNumber'] = full_df['Ticket'].str.extract(r'(\d+)').astype(float)
full_df['TicketNumber'] = full_df['TicketNumber'].fillna(0)

# Tạo đặc trưng từ prefix của ticket
full_df['TicketPrefix'] = full_df['Ticket'].str.extract('^([A-Za-z]+)')
full_df['TicketPrefix'] = full_df['TicketPrefix'].fillna('NUM')

# --- 2.11. Tạo đặc trưng từ Name (nâng cao) ---
# Độ dài tên
full_df['NameLength'] = full_df['Name'].str.len()

# Số từ trong tên
full_df['NameWords'] = full_df['Name'].str.split().str.len()

print("✅ Hoàn thành Feature Engineering siêu nâng cao.")

# BƯỚC 3: XỬ LÝ GIÁ TRỊ THIẾU NÂNG CAO CHO 'AGE'
print("🔄 Đang xử lý giá trị thiếu cho Age...")

temp_df = full_df.copy()
# Chuyển đổi categorical sang số
for col in ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 'Pclass_Sex', 'Pclass_Age', 'Sex_Age', 'TicketPrefix']:
    temp_df[col] = pd.factorize(temp_df[col])[0]

# Tạo AgeGroup và FareGroup
temp_df['AgeGroup'] = temp_df['Age'].apply(get_age_group)
temp_df['AgeGroup'] = pd.factorize(temp_df['AgeGroup'])[0]

temp_df['FareGroup'] = temp_df['Fare'].apply(get_fare_group)
temp_df['FareGroup'] = pd.factorize(temp_df['FareGroup'])[0]

# Features để dự đoán Age
features_for_age = ['Pclass', 'Fare', 'FamilySize', 'TicketFreq', 'HasCabin', 
                   'Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                   'Pclass_Sex', 'Pclass_Age', 'Sex_Age', 'TicketPrefix',
                   'FarePerPerson', 'NameLength', 'NameWords']

# Huấn luyện mô hình hồi quy
age_known = temp_df[temp_df['Age'].notna()]
age_unknown = temp_df[temp_df['Age'].isna()]

rfr = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=3, 
                           min_samples_leaf=1, random_state=42)
rfr.fit(age_known[features_for_age], age_known['Age'])

predicted_age = rfr.predict(age_unknown[features_for_age])
full_df.loc[full_df['Age'].isna(), 'Age'] = predicted_age

# Tạo AgeGroup và FareGroup sau khi đã điền Age
full_df['AgeGroup'] = full_df['Age'].apply(get_age_group)
full_df['FareGroup'] = full_df['Fare'].apply(get_fare_group)

print("✅ Đã điền giá trị thiếu cho Age bằng mô hình hồi quy siêu nâng cao.")

# BƯỚC 4: DỌN DẸP VÀ CHUẨN BỊ DỮ LIỆU
# Bỏ các cột không cần thiết
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
full_df = full_df.drop(columns=cols_to_drop)

# One-Hot Encoding cho tất cả categorical features
categorical_cols = ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                   'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Pclass_Age', 
                   'Sex_Age', 'TicketPrefix']
full_df = pd.get_dummies(full_df, columns=categorical_cols, drop_first=True)

# Tách lại thành tập train và test
train_final = full_df.iloc[:len(train_df)]
test_final = full_df.iloc[len(train_df):]

print("✅ Hoàn tất xử lý và dọn dẹp dữ liệu.")
print(f"📊 Kích thước tập train: {train_final.shape}")
print(f"📊 Kích thước tập test: {test_final.shape}")

# BƯỚC 5: FEATURE SELECTION
print("🔍 Thực hiện Feature Selection...")

# Sử dụng SelectKBest để chọn features quan trọng nhất
X_train = train_final
y_train = train_labels

# Chọn 30 features tốt nhất
selector = SelectKBest(score_func=f_classif, k=30)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(test_final)

# Lấy tên các features được chọn
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"✅ Đã chọn {len(selected_features)} features quan trọng nhất")

# BƯỚC 6: CHUẨN HÓA DỮ LIỆU
# Sử dụng RobustScaler thay vì StandardScaler (ít bị ảnh hưởng bởi outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# BƯỚC 7: HUẤN LUYỆN VÀ TỐI ƯU HÓA MÔ HÌNH (TỐI ƯU)
print("🤖 Bắt đầu huấn luyện mô hình (Phiên bản tối ưu)...")

# 1. Random Forest với GridSearchCV tối ưu (giảm tham số)
rf_params = {
    'n_estimators': [200, 300],  # Giảm từ 3 xuống 2
    'max_depth': [8, 12],        # Giảm từ 4 xuống 2
    'min_samples_split': [2, 5], # Giảm từ 3 xuống 2
    'min_samples_leaf': [1, 2]   # Giảm từ 3 xuống 2
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)  # Giảm CV từ 5 xuống 3
rf_grid.fit(X_train_selected, y_train)
print(f"✅ Random Forest - Best Score: {rf_grid.best_score_:.4f}")

# 2. Gradient Boosting với GridSearchCV tối ưu
gb_params = {
    'n_estimators': [100, 200],  # Giảm từ 3 xuống 2
    'learning_rate': [0.1, 0.15], # Giảm từ 4 xuống 2
    'max_depth': [3, 5]          # Giảm từ 4 xuống 2
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train_selected, y_train)
print(f"✅ Gradient Boosting - Best Score: {gb_grid.best_score_:.4f}")

# 3. AdaBoost với GridSearchCV tối ưu
ada_params = {
    'n_estimators': [50, 100],   # Giảm từ 3 xuống 2
    'learning_rate': [1.0, 1.5]  # Giảm từ 3 xuống 2
}

ada = AdaBoostClassifier(random_state=42)
ada_grid = GridSearchCV(ada, ada_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
ada_grid.fit(X_train_selected, y_train)
print(f"✅ AdaBoost - Best Score: {ada_grid.best_score_:.4f}")

# 4. Logistic Regression với GridSearchCV tối ưu
lr_params = {
    'C': [0.1, 1, 10],          # Giảm từ 4 xuống 3
    'penalty': ['l2']            # Chỉ dùng l2
}

lr = LogisticRegression(random_state=42, max_iter=1000)
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)
print(f"✅ Logistic Regression - Best Score: {lr_grid.best_score_:.4f}")

# 5. SVM với GridSearchCV tối ưu
svm_params = {
    'C': [1, 10],               # Giảm từ 4 xuống 2
    'gamma': ['scale', 'auto'], # Giảm từ 6 xuống 2
    'kernel': ['rbf']           # Chỉ dùng rbf
}

svm = SVC(random_state=42, probability=True)
svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
svm_grid.fit(X_train_scaled, y_train)
print(f"✅ SVM - Best Score: {svm_grid.best_score_:.4f}")

# 6. KNN với GridSearchCV tối ưu
knn_params = {
    'n_neighbors': [3, 5, 7],   # Giảm từ 5 xuống 3
    'weights': ['distance']     # Chỉ dùng distance
}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
knn_grid.fit(X_train_scaled, y_train)
print(f"✅ KNN - Best Score: {knn_grid.best_score_:.4f}")

# BƯỚC 8: ENSEMBLE MODEL NÂNG CAO
print("🎯 Tạo Ensemble Model nâng cao...")

# Tạo Voting Classifier với các mô hình tốt nhất
ensemble = VotingClassifier([
    ('rf', rf_grid.best_estimator_),
    ('gb', gb_grid.best_estimator_),
    ('ada', ada_grid.best_estimator_),
    ('lr', lr_grid.best_estimator_),
    ('svm', svm_grid.best_estimator_),
    ('knn', knn_grid.best_estimator_)
], voting='soft')

# Huấn luyện ensemble
ensemble.fit(X_train_selected, y_train)

# Đánh giá ensemble
ensemble_score = cross_val_score(ensemble, X_train_selected, y_train, cv=3).mean()
print(f"🎯 Ensemble Model - CV Score: {ensemble_score:.4f}")

# BƯỚC 9: DỰ ĐOÁN VÀ TẠO FILE SUBMISSION
print("📝 Tạo file submission...")

# Dự đoán với ensemble model
predictions = ensemble.predict(X_test_selected)

# Tạo file submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

# Lưu file
submission.to_csv('submission_advanced_optimized.csv', index=False)

print("🎉 Đã tạo file submission_advanced_optimized.csv thành công!")
print(f"📊 Dự đoán: {predictions.sum()} người sống sót trong {len(predictions)} hành khách test")

# BƯỚC 10: HIỂN THỊ TẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG

print("\n📈 Tầm quan trọng của các đặc trưng (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))
print("🚢 Kết thúc xử lý dữ liệu Titanic (Phiên bản Advanced Tối Ưu).")