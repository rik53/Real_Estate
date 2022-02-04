Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy API: flask Данные: с kaggle - https://www.kaggle.com/c/real-estate-price-prediction-moscow/data

Задача: предсказать цены на квартиры(поле Price). Задача регрессии.

Используемые признаки:
Square (float)
Rooms (float)
Social_3 (int)
Social_1 (int)


Модель: GradientBoostingRegressor

Клонируем репозиторий и создаем образ
$ git clone https://github.com/rik53/Real_Estate.git
$ cd Real_Estate
$ docker build -t Real_Estate .
Запускаем контейнер
$ docker run -d -p 8180:8180 -p 8181:8181 -v <your_local_path_to_pretrained_models>:/app/app/models Real_Estate