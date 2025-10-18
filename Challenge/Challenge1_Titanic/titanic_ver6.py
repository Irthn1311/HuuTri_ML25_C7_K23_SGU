import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

print("🚢 Titanic ML - Phiên bản Ver6 (Ultra Simple + Focused + Optimized)")
print("🎯 Mục tiêu: Vượt qua 0.78468 với approach đơn giản tối ưu!")

# =============================================================================
# BƯỚC 1: TẢI DỮ LIỆU RIÊNG BIỆT
# =============================================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_labels = train_df['Survived']
test_ids = test_df['PassengerId']

print("✅ Tải dữ liệu train và test riêng biệt thành công.")
print(f"📊 Train shape: {train_df.shape}")
print(f"📊 Test shape: {test_df.shape}")

# =============================================================================
# BƯỚC 2: HÀM FEATURE ENGINEERING "ULTRA SIMPLE + FOCUSED"
# =============================================================================
def create_features_ultra_simple(df):
    """Tạo features siêu đơn giản - chỉ những gì thực sự quan trọng nhất"""
    df = df.copy()
    
    # --- 2.1. Xử lý giá trị thiếu cơ bản (cố định) ---
    df['Embarked'] = df['Embarked'].fillna('S')  # Mode cố định
    df['Fare'] = df['Fare'].fillna(7.91)  # Median cố định
    
    # --- 2.2. Deck từ Cabin (QUAN TRỌNG NHẤT) ---
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    
    # --- 2.3. Title từ Name (QUAN TRỌNG NHẤT) ---
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Gộp thông minh - chỉ giữ những gì quan trọng
    title_mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    # --- 2.4. Family Features (QUAN TRỌNG) ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['LargeFamily'] = (df['FamilySize'] > 4).astype(int)
    
    # --- 2.5. Ticket Features (QUAN TRỌNG) ---
    df['TicketFreq'] = df['Ticket'].map(df['Ticket'].value_counts())
    df['TicketGroup'] = df['TicketFreq'].apply(lambda x: 'Single' if x == 1 else 'Group')
    
    # --- 2.6. Age Group (QUAN TRỌNG) ---
    def get_age_group(age):
        if pd.isna(age): return 'Unknown'
        elif age < 1: return 'Infant'
        elif age < 16: return 'Child'
        elif age < 30: return 'Young_Adult'
        elif age < 50: return 'Adult'
        else: return 'Senior'
    
    df['AgeGroup'] = df['Age'].apply(get_age_group)
    
    # --- 2.7. Fare Group (QUAN TRỌNG) ---
    def get_fare_group(fare):
        if fare <= 7.91: return 'Low'
        elif fare <= 14.45: return 'Medium'
        elif fare <= 31: return 'High'
        else: return 'Very_High'
    
    df['FareGroup'] = df['Fare'].apply(get_fare_group)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # --- 2.8. Interaction Features (QUAN TRỌNG NHẤT) ---
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['Sex_Age'] = df['Sex'] + '_' + df['Age'].apply(lambda x: 'Child' if x < 16 else 'Adult')
    
    # --- 2.9. Cabin Features (QUAN TRỌNG) ---
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    return df

# =============================================================================
# BƯỚC 3: XỬ LÝ DỮ LIỆU TRAIN VÀ TEST RIÊNG BIỆT
# =============================================================================
print("🔄 Xử lý dữ liệu train và test riêng biệt...")

# Xử lý train set
train_processed = create_features_ultra_simple(train_df)

# Xử lý test set
test_processed = create_features_ultra_simple(test_df)

print("✅ Hoàn thành xử lý dữ liệu train và test riêng biệt.")

# =============================================================================
# BƯỚC 4: XỬ LÝ MISSING VALUES CHO AGE (CHỈ TRÊN TRAIN SET)
# =============================================================================
print("🔄 Xử lý missing values cho Age...")

# Tạo features để dự đoán Age
temp_train = train_processed.copy()
for col in ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Sex_Age']:
    temp_train[col] = pd.factorize(temp_train[col])[0]

# Features để dự đoán Age (chỉ những gì quan trọng)
features_for_age = ['Pclass', 'Fare', 'FamilySize', 'TicketFreq', 'HasCabin', 
                   'Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                   'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Sex_Age', 'FarePerPerson']

# Huấn luyện mô hình hồi quy CHỈ trên train set
age_known = temp_train[temp_train['Age'].notna()]
age_unknown = temp_train[temp_train['Age'].isna()]

if len(age_unknown) > 0:
    rfr = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5, 
                               min_samples_leaf=2, random_state=42)
    rfr.fit(age_known[features_for_age], age_known['Age'])
    predicted_age = rfr.predict(age_unknown[features_for_age])
    train_processed.loc[train_processed['Age'].isna(), 'Age'] = predicted_age

# Sử dụng mô hình đã train để dự đoán Age cho test
if len(test_processed[test_processed['Age'].isna()]) > 0:
    temp_test = test_processed.copy()
    for col in ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Sex_Age']:
        temp_test[col] = pd.factorize(temp_test[col])[0]
    
    age_missing_mask = test_processed['Age'].isna()
    predicted_age_test = rfr.predict(temp_test[age_missing_mask][features_for_age])
    test_processed.loc[age_missing_mask, 'Age'] = predicted_age_test

print("✅ Hoàn thành xử lý missing values cho Age.")

# =============================================================================
# BƯỚC 5: CHUẨN BỊ DỮ LIỆU CHO PIPELINE
# =============================================================================
# Bỏ các cột không cần thiết
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train_clean = train_processed.drop(columns=cols_to_drop + ['Survived'])
test_clean = test_processed.drop(columns=cols_to_drop)

# Định nghĩa các cột
categorical_cols = ['Sex', 'Embarked', 'Deck', 'Title', 'TicketGroup', 
                   'AgeGroup', 'FareGroup', 'Pclass_Sex', 'Sex_Age']
numerical_cols = [col for col in train_clean.columns if col not in categorical_cols]

print(f"📊 Categorical columns: {len(categorical_cols)}")
print(f"📊 Numerical columns: {len(numerical_cols)}")

# =============================================================================
# BƯỚC 6: TẠO PIPELINE "ULTRA SIMPLE + FOCUSED"
# =============================================================================
print("🔧 Tạo Pipeline Ultra Simple + Focused...")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Tạo pipeline cho RandomForest (Ultra Conservative)
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(score_func=f_classif, k=20)),  # Chỉ 20 features tốt nhất
    ('scaler', RobustScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("✅ Hoàn thành tạo Pipeline.")

# =============================================================================
# BƯỚC 7: GRID SEARCH ULTRA CONSERVATIVE
# =============================================================================
print("🤖 Thực hiện Grid Search Ultra Conservative...")

# Random Forest parameters (Ultra Conservative - tránh overfitting)
rf_params = {
    'classifier__n_estimators': [200, 300, 400],
    'classifier__max_depth': [5, 6, 7],  # Rất conservative
    'classifier__min_samples_split': [10, 15],  # Rất conservative
    'classifier__min_samples_leaf': [3, 5],  # Rất conservative
    'classifier__max_features': ['sqrt', 'log2']  # Tránh overfitting
}

# Grid Search với Cross Validation
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(train_clean, train_labels)

print(f"✅ RandomForest - Best Score: {rf_grid.best_score_:.4f}")
print(f"✅ Best Parameters: {rf_grid.best_params_}")

# =============================================================================
# BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH
# =============================================================================
print("📊 Đánh giá mô hình...")

# Cross-validation với StratifiedKFold
cv_scores = cross_val_score(rf_grid.best_estimator_, train_clean, train_labels, cv=5)
print(f"📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# BƯỚC 9: DỰ ĐOÁN VÀ TẠO FILE SUBMISSION
# =============================================================================
print("📝 Tạo file submission...")

# Dự đoán với best model
predictions = rf_grid.best_estimator_.predict(test_clean)

# Tạo file submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

# Lưu file
submission.to_csv('submission_titanic_ver6.csv', index=False)

print("🎉 Đã tạo file submission_titanic_ver6.csv thành công!")
print(f"📊 Dự đoán: {predictions.sum()} người sống sót trong {len(predictions)} hành khách test")

# =============================================================================
# BƯỚC 10: HIỂN THỊ THÔNG TIN MÔ HÌNH
# =============================================================================
print("\n📈 Thông tin mô hình:")
print(f"🎯 CV Score: {cv_scores.mean():.4f}")
print(f"🎯 Best Parameters: {rf_grid.best_params_}")
print(f"🎯 Predictions: {predictions.sum()}/{len(predictions)} survivors ({predictions.sum()/len(predictions)*100:.1f}%)")

# =============================================================================
# BƯỚC 11: PHÂN TÍCH FEATURE IMPORTANCE
# =============================================================================
print("\n📈 Feature Importance Analysis:")

try:
    # Lấy feature names từ pipeline
    feature_names = numerical_cols + list(rf_grid.best_estimator_.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols))
    selected_features = feature_names[rf_grid.best_estimator_.named_steps['selector'].get_support()]

    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_grid.best_estimator_.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 15 features quan trọng nhất:")
    print(feature_importance.head(15))
except:
    print("Không thể hiển thị feature importance do lỗi indexing.")

print("\n🚢 Kết thúc - Titanic Ver6 (Ultra Simple + Focused + Optimized)")
print("💡 Mục tiêu: Vượt qua 0.78468 với approach đơn giản tối ưu!")
print("🎯 Approach: Ultra simple features + Ultra conservative parameters + Single model")
