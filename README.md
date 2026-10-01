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

<img width="797" height="608" alt="Screenshot_2" src="https://github.com/user-attachments/assets/a0cdd072-8a8a-4594-88e4-00a7ff418df3" />
<img width="698" height="592" alt="Screenshot_1" src="https://github.com/user-attachments/assets/38f3dea0-fbd4-4ea4-9fcf-8d108f85529b" />


