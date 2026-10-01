# ML-model-insurance-company

РУ:
Образовательная практика по поиску наиболее сильных характеристик из готовых данных для обучения модели ML на основе данных страховой компании

EN:
Educational practice on finding the strongest features from ready-made data for training an ML model based on data from an insurance company

-----------Описание-----------

РУ: 
Это учебный проект для отработки персональных навыков в сфере аналитики данных. 
Нахождение сильных признаков для ML модели на основе уже существующих признаков, 
с целью научить модель предсказывать предполагаемые расходы для нового клиента и помогать Страховой компании на корню понимать сумму для конкретного человека.

EN:
This is a training project for developing personal skills in the field of data analytics. 
Finding strong features for an ML model based on pre-existing features in order to teach the model,
to predict estimated costs for a new client and help the Insurance Company understand the amount for a particular person at the root

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
1. smoker_obese (~0.83) (признак - курящий с ИМТ > 30)
2. smoker_bmi   (~0.10) (признак - курящие с разделением на ИМТ для выборки сильнейшей связки)

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


