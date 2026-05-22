# Лабораторная работа №9
## Анализ успеваемости студентов

#[Переход в репозиторий лабораторной](https://sourcecraft.dev/nyabarykina/itmo-python-lab-template)

## Данные
- **Источник:** `StudentsPerformance.csv` (1000 записей, 8 столбцов)
- **Числовые признаки:** math_score, reading_score, writing_score
- **Категориальные признаки:** gender, race_ethnicity, parental_education, lunch, test_prep

## Предобработка
- Переименованы столбцы (убраны пробелы, спецсимволы)
- Созданы новые признаки: `average_score`, `total_score`, `performance_level`
- Пропуски отсутствуют

## Ключевые результаты  
1) test_prep (completed): +7.6 балла  
2) lunch (standard): +8.6 балла  
3) parental_education (master's): +11 баллов vs some high school  
4) gender (female): +3.8 балла(в среднем)  
5) race_ethnicity (group E vs A): +9.8 балла  

## Визуализации
1. Гистограмма распределения math_score
2. Столбчатые диаграммы по test_prep и gender
3. Диаграмма рассеяния (math_score vs reading_score)
4. Доп: Boxplot по parental_education

## CI/CD
- **Платформа:** SourceCraft
- **Автоматизация:** установка зависимостей -> выполнение ноутбука (papermill) -> генерация HTML-отчета
- **Артефакты:** `executed_lab.ipynb`, `report.html`
- **Маркер выполнения:** в ноутбук добавляется Markdown-ячейка с подтверждением запуска в CI/CD (время, hostname)

## Вывод
Успеваемость сильно коррелирует с социально-экономическими факторами: подготовкой, питанием, образованием родителей. CI/CD пайплайн полностью автоматизирует процесс выполнения и формирования отчета.