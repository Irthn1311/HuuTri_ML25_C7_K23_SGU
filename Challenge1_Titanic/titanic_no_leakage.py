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


# BƯỚC 1: TẢI DỮ LIỆU RIÊNG BIỆT
print("🚢 Bắt đầu xử lý dữ liệu Titanic (Phiên bản KHÔNG Data Leakage)...")

# Tải dữ liệu
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Lưu lại các thông tin cần thiết
train_labels = train_df['Survived']
test_ids = test_df['PassengerId']

print("✅ Tải dữ liệu thành công.")
print(f"📊 Kích thước tập train: {train_df.shape}")
print(f"📊 Kích thước tập test: {test_df.shape}")

# BƯỚC 2: TẠO CÁC HÀM XỬ LÝ FEATURE ENGINEERING
def get_age_group(age):
    """Phân loại nhóm tuổi"""
    if pd.isna(age): return 'Unknown'
    elif age < 1: return 'Infant'
    elif age < 5: return 'Toddler'
    elif age < 12: return 'Child'
    elif age < 18: return 'Teen'
    elif age < 30: return 'Young_Adult'
    elif age < 50: return 'Adult'
    elif age < 65: return 'Middle_Age'
    else: return 'Senior'

def get_fare_group(fare):
    """Phân loại nhóm giá vé"""
    if fare <= 7.91: return 'Low'
    elif fare <= 14.45: return 'Medium'
    elif fare <= 31: return 'High'
    else: return 'Very_High'

def process_features(df, is_train=True, ticket_freq_dict=None, title_mapping=None):
    """
    Xử lý feature engineering cho một dataset
    Args:
        df: DataFrame cần xử lý
        is_train: True nếu là train data, False nếu là test data
        ticket_freq_dict: Dictionary chứa tần suất ticket (chỉ dùng cho test)
        title_mapping: Dictionary mapping title (chỉ dùng cho test)
    """
    df_processed = df.copy()
    
    # --- 2.1. Xử lý giá trị thiếu cơ bản ---
    if is_train:
        # Chỉ sử dụng thông tin từ train data
        most_frequent_port = df_processed['Embarked'].mode()[0]
        median_fare = df_processed['Fare'].median()
    else:
        # Sử dụng thông tin đã tính từ train data
        most_frequent_port = 'S'  # Giá trị mặc định
        median_fare = 14.45  # Giá trị mặc định
    
    df_processed['Embarked'] = df_processed['Embarked'].fillna(most_frequent_port)
    df_processed['Fare'] = df_processed['Fare'].fillna(median_fare)
    
    # --- 2.2. Tạo đặc trưng 'Deck' từ 'Cabin' ---
    df_processed['Deck'] = df_processed['Cabin'].str[0].fillna('U')
    
    # --- 2.3. Tạo đặc trưng 'Title' từ 'Name' ---
    df_processed['Title'] = df_processed['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    if is_train:
        # Tạo mapping từ train data
        title_mapping = {
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
        }
        df_processed['Title'] = df_processed['Title'].replace(title_mapping)
    else:
        # Sử dụng mapping từ train data
        df_processed['Title'] = df_processed['Title'].replace(title_mapping)
    
    # --- 2.4. Tạo các đặc trưng Family ---
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    df_processed['SmallFamily'] = ((df_processed['FamilySize'] >= 2) & (df_processed['FamilySize'] <= 4)).astype(int)
    df_processed['LargeFamily'] = (df_processed['FamilySize'] > 4).astype(int)
    
    # --- 2.5. Tạo đặc trưng Ticket ---
    if is_train:
        # Tính ticket frequency từ train data
        ticket_counts = df_processed['Ticket'].value_counts()
        ticket_freq_dict = ticket_counts.to_dict()
    
    df_processed['TicketFreq'] = df_processed['Ticket'].map(ticket_freq_dict).fillna(1)
    df_processed['TicketGroup'] = df_processed['TicketFreq'].apply(lambda x: 'Single' if x == 1 else 'Group')
    
    # --- 2.6. Tạo Fare per person ---
    df_processed['FarePerPerson'] = df_processed['Fare'] / df_processed['FamilySize']
    
    # --- 2.7. Tạo các đặc trưng tương tác quan trọng ---
    df_processed['Pclass_Sex'] = df_processed['Pclass'].astype(str) + '_' + df_processed['Sex']
    df_processed['Pclass_Age'] = df_processed['Pclass'].astype(str) + '_' + df_processed['Age'].apply(lambda x: 'Child' if x < 16 else 'Adult')
    df_processed['Sex_Age'] = df_processed['Sex'] + '_' + df_processed['Age'].apply(lambda x: 'Child' if x < 16 else 'Adult')
    
    # --- 2.8. Tạo đặc trưng Cabin ---
    df_processed['HasCabin'] = df_processed['Cabin'].notna().astype(int)
    df_processed['CabinNumber'] = df_processed['Cabin'].str.extract(r'(\d+)').astype(float)
    df_processed['CabinNumber'] = df_processed['CabinNumber'].fillna(0)
    
    # --- 2.9. Tạo đặc trưng từ Ticket ---
    df_processed['TicketNumber'] = df_processed['Ticket'].str.extract(r'(\d+)').astype(float)
    df_processed['TicketNumber'] = df_processed['TicketNumber'].fillna(0)
    
    df_processed['TicketPrefix'] = df_processed['Ticket'].str.extract('^([A-Za-z]+)')
    df_processed['TicketPrefix'] = df_processed['TicketPrefix'].fillna('NUM')
    
    # --- 2.10. Tạo đặc trưng từ Name ---
    df_processed['NameLength'] = df_processed['Name'].str.len()
    df_processed['NameWords'] = df_processed['Name'].str.split().str.len()
    
    return df_processed, ticket_freq_dict, title_mapping

# BƯỚC 3: XỬ LÝ TRAIN DATA
print("🔄 Xử lý Train Data...")
train_processed, ticket_freq_dict, title_mapping = process_features(train_df, is_train=True)

# BƯỚC 4: XỬ LÝ MISSING VALUES CHO AGE TRONG TRAIN DATA
print("🔄 Xử lý missing values cho Age trong Train Data...")

# Tạo temp dataframe cho việc imputation
temp_train = train_processed.copy()

# Chuyển đổi categorical sang số
categorical_cols = ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 'Pclass_Sex', 'Pclass_Age', 'Sex_Age', 'TicketPrefix']
for col in categorical_cols:
    temp_train[col] = pd.factorize(temp_train[col])[0]

# Tạo AgeGroup và FareGroup
temp_train['AgeGroup'] = temp_train['Age'].apply(get_age_group)
temp_train['AgeGroup'] = pd.factorize(temp_train['AgeGroup'])[0]

temp_train['FareGroup'] = temp_train['Fare'].apply(get_fare_group)
temp_train['FareGroup'] = pd.factorize(temp_train['FareGroup'])[0]

# Features để dự đoán Age
features_for_age = ['Pclass', 'Fare', 'FamilySize', 'TicketFreq', 'HasCabin', 
                   'Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                   'Pclass_Sex', 'Pclass_Age', 'Sex_Age', 'TicketPrefix',
                   'FarePerPerson', 'NameLength', 'NameWords']

# Huấn luyện mô hình hồi quy cho Age
age_known = temp_train[temp_train['Age'].notna()]
age_unknown = temp_train[temp_train['Age'].isna()]

if len(age_unknown) > 0:
    rfr = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=3, 
                               min_samples_leaf=1, random_state=42)
    rfr.fit(age_known[features_for_age], age_known['Age'])
    
    predicted_age = rfr.predict(age_unknown[features_for_age])
    train_processed.loc[train_processed['Age'].isna(), 'Age'] = predicted_age

# Tạo AgeGroup và FareGroup sau khi đã điền Age
train_processed['AgeGroup'] = train_processed['Age'].apply(get_age_group)
train_processed['FareGroup'] = train_processed['Fare'].apply(get_fare_group)

print("✅ Hoàn thành xử lý Train Data.")

# BƯỚC 5: XỬ LÝ TEST DATA
print("🔄 Xử lý Test Data...")
test_processed, _, _ = process_features(test_df, is_train=False, 
                                       ticket_freq_dict=ticket_freq_dict, 
                                       title_mapping=title_mapping)

# BƯỚC 6: XỬ LÝ MISSING VALUES CHO AGE TRONG TEST DATA
print("🔄 Xử lý missing values cho Age trong Test Data...")

# Sử dụng mô hình đã huấn luyện từ train data
temp_test = test_processed.copy()

# Chuyển đổi categorical sang số (sử dụng cùng mapping như train)
for col in categorical_cols:
    temp_test[col] = pd.factorize(temp_test[col])[0]

# Tạo AgeGroup và FareGroup
temp_test['AgeGroup'] = temp_test['Age'].apply(get_age_group)
temp_test['AgeGroup'] = pd.factorize(temp_test['AgeGroup'])[0]

temp_test['FareGroup'] = temp_test['Fare'].apply(get_fare_group)
temp_test['FareGroup'] = pd.factorize(temp_test['FareGroup'])[0]

# Dự đoán Age cho test data
age_unknown_test = temp_test[temp_test['Age'].isna()]
if len(age_unknown_test) > 0:
    predicted_age_test = rfr.predict(age_unknown_test[features_for_age])
    test_processed.loc[test_processed['Age'].isna(), 'Age'] = predicted_age_test

# Tạo AgeGroup và FareGroup sau khi đã điền Age
test_processed['AgeGroup'] = test_processed['Age'].apply(get_age_group)
test_processed['FareGroup'] = test_processed['Fare'].apply(get_fare_group)

print("✅ Hoàn thành xử lý Test Data.")

# BƯỚC 7: DỌN DẸP VÀ CHUẨN BỊ DỮ LIỆU
print("🔄 Dọn dẹp và chuẩn bị dữ liệu...")

# Bỏ các cột không cần thiết
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train_clean = train_processed.drop(columns=cols_to_drop)
test_clean = test_processed.drop(columns=cols_to_drop)

# Đảm bảo loại bỏ target variable khỏi features
if 'Survived' in train_clean.columns:
    train_clean = train_clean.drop(columns=['Survived'])

# One-Hot Encoding cho tất cả categorical features
categorical_cols_final = ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                         'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Pclass_Age', 
                         'Sex_Age', 'TicketPrefix']

# Lấy tất cả unique values từ train để đảm bảo consistency
all_categories = {}
for col in categorical_cols_final:
    all_categories[col] = train_clean[col].unique()

# One-hot encoding cho train
train_final = pd.get_dummies(train_clean, columns=categorical_cols_final, drop_first=True)

# One-hot encoding cho test với cùng columns
test_final = pd.get_dummies(test_clean, columns=categorical_cols_final, drop_first=True)

# Đảm bảo test có cùng columns với train
missing_cols = set(train_final.columns) - set(test_final.columns)
for col in missing_cols:
    test_final[col] = 0

# Sắp xếp columns theo thứ tự giống train
test_final = test_final[train_final.columns]

print("✅ Hoàn tất dọn dẹp và chuẩn bị dữ liệu.")
print(f"📊 Kích thước tập train: {train_final.shape}")
print(f"📊 Kích thước tập test: {test_final.shape}")

# BƯỚC 8: FEATURE SELECTION
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

# BƯỚC 9: CHUẨN HÓA DỮ LIỆU
# Sử dụng RobustScaler thay vì StandardScaler (ít bị ảnh hưởng bởi outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# BƯỚC 10: HUẤN LUYỆN VÀ TỐI ƯU HÓA MÔ HÌNH
print("🤖 Bắt đầu huấn luyện mô hình...")

# 1. Random Forest với GridSearchCV
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [8, 12],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_selected, y_train)
print(f"✅ Random Forest - Best Score: {rf_grid.best_score_:.4f}")

# 2. Gradient Boosting với GridSearchCV
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.15],
    'max_depth': [3, 5]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train_selected, y_train)
print(f"✅ Gradient Boosting - Best Score: {gb_grid.best_score_:.4f}")

# 3. AdaBoost với GridSearchCV
ada_params = {
    'n_estimators': [50, 100],
    'learning_rate': [1.0, 1.5]
}

ada = AdaBoostClassifier(random_state=42)
ada_grid = GridSearchCV(ada, ada_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
ada_grid.fit(X_train_selected, y_train)
print(f"✅ AdaBoost - Best Score: {ada_grid.best_score_:.4f}")

# 4. Logistic Regression với GridSearchCV
lr_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

lr = LogisticRegression(random_state=42, max_iter=1000)
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)
print(f"✅ Logistic Regression - Best Score: {lr_grid.best_score_:.4f}")

# 5. SVM với GridSearchCV
svm_params = {
    'C': [1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

svm = SVC(random_state=42, probability=True)
svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
svm_grid.fit(X_train_scaled, y_train)
print(f"✅ SVM - Best Score: {svm_grid.best_score_:.4f}")

# 6. KNN với GridSearchCV
knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['distance']
}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
knn_grid.fit(X_train_scaled, y_train)
print(f"✅ KNN - Best Score: {knn_grid.best_score_:.4f}")

# BƯỚC 11: ENSEMBLE MODEL
print("🎯 Tạo Ensemble Model...")

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

# BƯỚC 12: DỰ ĐOÁN VÀ TẠO FILE SUBMISSION
print("📝 Tạo file submission...")

# Dự đoán với ensemble model
predictions = ensemble.predict(X_test_selected)

# Tạo file submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

# Lưu file
submission.to_csv('submission_no_leakage_ver3.csv', index=False)

print("🎉 Đã tạo file submission_no_leakage_ver3.csv thành công!")
print(f"📊 Dự đoán: {predictions.sum()} người sống sót trong {len(predictions)} hành khách test")

# BƯỚC 13: HIỂN THỊ TẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG
print("\n📈 Tầm quan trọng của các đặc trưng (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))
print("🚢 Kết thúc xử lý dữ liệu Titanic (Phiên bản KHÔNG Data Leakage).")
