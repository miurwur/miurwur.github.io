# Лабораторная работа №4  
## Цель работы  
Заполнить борд, рассмотреть модели для решения задачи классификации. Оценить возможность достижения ROC-AUC > 0.9 и обосновать полученные результаты.

#[Переход на блокнот](https://colab.research.google.com/drive/13KDc8E0TlWDncBpdfIxno-kAGYjbghdT?usp=sharing)

## Ход Работы  
### 1. Анализ исходных данных  
1.1 Характеристики датасета
- <img src="/img/labs_4_5/ds_info.png" alt="ds_info" width="300">
2.2 Обработка данных
Заполняем пропуски средними значениями
```python
train_mean = training_data.mean()
training_data.fillna(train_mean, inplace = True)
```
