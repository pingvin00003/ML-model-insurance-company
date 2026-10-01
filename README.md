# ML-model-insurance-company

Machine Learning project for predicting individual medical insurance costs.

РУ:
Проект исследует разработку функций Random Forest regression. 
Выявить наиболее сильные факторы, влияющие на страховые расходы.

EN:
The project explores feature engineering and Random Forest regression
to identify the strongest factors affecting insurance charges.

## 📊 Model Results

| Metric | Result |
|---|---:|
| Model | Random Forest Regressor |
| R² | **0.846** |
| MAE | **2776** |
| Train / Test | **80% / 20%** |


-----------Описание-----------

РУ: 
Это учебный проект для отработки персональных навыков в сфере аналитики данных. 
Нахождение сильных признаков для ML модели на основе уже существующих признаков.
Модель предназначена для прогнозирования индивидуальных страховых расходов на основе демографических и поведенческих характеристик клиента

EN:


P.S: 
РУ:
Модель нуждается в дополнительном до обучении так как признак smoker_obese насчитывает >200 записей на общее ~1300 из за данного разброса модель может показывать не точности.

EN:
The model needs additional pre-training since the smoker_obese attribute has >200 entries for a total of ~1300 due to this spread, the model may not show accuracy.

-------------------------------


---Признаки---
- age, bmi, children, sex, smoker, region
- Найденные: smoker_bmi, age_smoker, is_obese, smoker_obese, bmi_squared

---Модель---
```python
RandomForestRegressor (n_estimators=500, max_depth=15, min_samples_leaf=2)
```

---Метрики (обучение-80%, тест-20%)---
- R²  = 0.846
- MAE = 2776

---Главные признаки---
1. smoker_obese (~0.83) (признак - бинарный признак, равный 1, если клиент одновременно является курильщиком и имеет BMI > 30)
   Этот признак оказался наиболее значимым в модели и отражает сочетание двух факторов риска
3. smoker_bmi   (~0.10) (признак - курящие с разделением на ИМТ для выборки сильнейшей связки)

---Использование---
```python
import joblib, pandas as pd
model = joblib.load('charges_model.pkl')
features = joblib.load('charges_features.pkl')
model.predict(new_df[features])
```

Числовые показатели:
<img width="1091" height="688" alt="Screenshot_3" src="https://github.com/user-attachments/assets/384b8c0f-4b3b-46a3-9e9a-a3dec725617b" />
<img width="469" height="73" alt="Screenshot_4" src="https://github.com/user-attachments/assets/7924b83c-89a1-4e91-8f32-303f7532e313" />

Предсказание нового клиента:
```python
new_client = pd.DataFrame({
    'age': [35], 'bmi': [33], 'children': [2],
    'sex_num': [1], 'smoker_num': [1], 'smoker_bmi': [33]
})
```
<img width="188" height="58" alt="Screenshot_5" src="https://github.com/user-attachments/assets/cf0f123c-8153-4cd1-aad1-c8322075f0fd" />

Работа модели:


<img width="797" height="608" alt="Screenshot_2" src="https://github.com/user-attachments/assets/a0cdd072-8a8a-4594-88e4-00a7ff418df3" />
<img width="698" height="592" alt="Screenshot_1" src="https://github.com/user-attachments/assets/38f3dea0-fbd4-4ea4-9fcf-8d108f85529b" />


