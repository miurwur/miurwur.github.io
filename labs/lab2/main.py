# 
# Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy

# Формат выполнения: самостоятельная работа.

# Перед началом:
# 1. Создайте виртуальное окружение:
#    python -m venv numpy_env
   
# 2. Активируйте виртуальное окружение:
#    - Windows: numpy_env\Scripts\activate
#    - Linux/Mac: source numpy_env/bin/activate
   
# 3. Установите зависимости:
#    pip install numpy matplotlib seaborn pandas pytest

# Структура проекта:

# numpy_lab/
# ├── main.py
# ├── test.py
# ├── data/
# │   └── students_scores.csv
# └── plots/

# В папке data создайте файл students_scores.csv со следующим содержимым:

# math,physics,informatics
# 78,81,90
# 85,89,88
# 92,94,95
# 70,75,72
# 88,84,91
# 95,99,98
# 60,65,70
# 73,70,68
# 84,86,85
# 90,93,92

# (Дополнительно можно использовать публичные датасеты Kaggle:
# Students Performance Dataset:
# https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
# или любой аналогичный табличный CSV)

# Задача: реализовать все функции, чтобы проходили тесты.
# 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    
    Example:
        >>> create_vector()
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # Подсказка: используйте np.arange(10)

    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Изучить:
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1

    Example:
    >>> create_matrix()
    [[0.12494914 0.13225594 0.79438915 0.33566234 0.49417778]
    [0.79833819 0.6527756  0.52481415 0.55842081 0.28112532]
    [0.60251706 0.18195474 0.80991619 0.14940745 0.68856513]
    [0.18727448 0.98386592 0.71418467 0.37288643 0.58184503]
    [0.11892152 0.74578275 0.63338907 0.57506871 0.24403581]]
    """
    # Подсказка: используйте np.random.rand(5,5)

    return np.random.rand(5,5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5)

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    
    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)

    Example:
    >>> vec = np.arrange(10)
    >>> reshape_vector(vec).shape
    (2, 5)
    """
    # Подсказка: используйте vec.reshape(2,5)
    
    if vec.shape != (10,):
        raise ValueError("Входной массив должен иметь форму (10,)")
   
    return(np.reshape(vec, (2,5)))



def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    
    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица

    Example:
    >>> mat = np.array([[1,2],[3,4]])
    >>> transpose_matrix(mat)
        array([[1, 3],
               [2, 4]])
    """
    # Подсказка: используйте mat.T или np.transpose(mat)

    return np.transpose(mat)


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения

    Example:
    >>> vector_add(np.array([1,2]), np.array([3,4]))
    array([4,6])
    """
    # Подсказка: используйте оператор +
    
    if a.shape != b.shape:
        raise ValueError("Векторы должны быть одинаковой длины")
    
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: int | float) -> np.ndarray:
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    
    Example:
    >>> scalar_multiply(np.array([1,2]), 2)
    array([2,4])
    """
    # Подсказка: используйте оператор *
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    
    Example:
    >>> elementwise_multiply(np.array([1,2]), np.array([3,4]))
    array([3,8])
    """
    # Подсказка: используйте оператор *
    if a.shape != b.shape:
        raise ValueError("Векторы/матрицы должны иметь одинаковую форму")
    
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    
    Example:
    >>> dot_product(np.array([1, 2]), np.array([3, 4]))
    11
    """
    # Подсказка: используйте np.dot(a, b)

    if a.shape != b.shape:
        raise ValueError("Векторы должны иметь одинаковую длину")
    
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    
    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    
    Example:
    >>> A = np.array([1,2], [3,4])
    >>> B = np.array([2,0], [1,2])
    >>> matrix_multipy(A,B)
    [[ 4  4]
    [10  8]]
    """
    # Подсказка: используйте a @ b или np.matmul(a, b)

    if a.shape[1] != b.shape[0]:
        raise ValueError("Количество столбцов первой матрицы должно равняться количеству строк второй")
    
    return np.matmul(a, b)


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы

    Example:
    >>> A = np.array([[1,2], [3,4]])
    >>> matrix_determinant(A)
    -2.0
    """
    # Подсказка: используйте np.linalg.det(a)

    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    
    return np.linalg.det(a)
    


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    
    Example:
    >>> A = np.array([[1,2], [3,4]])
    >>> matrix_inverse(A)
    [[-2.   1. ]
    [ 1.5 -0.5]]
    """
    # Подсказка: используйте np.linalg.inv(a)

    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    if np.linalg.det(a) == 0:
        raise ValueError("Матрица вырождена, обратной не существует")
    
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    
    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x

    Example:
        >>> A = np.array([[2, 1], [1, 3]])
        >>> b = np.array([1, 2])
        >>> solve_linear_system(A, b)
        array([0.2, 0.6])
    """
    # Подсказка: используйте np.linalg.solve(a, b)
    
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица коэффициентов должна быть квадратной")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Размерности матрицы коэффициентов и вектора свободных членов несовместимы")
    if np.linalg.det(a) == 0:
        raise ValueError("Матрица коэффициентов вырождена, система не имеет единственного решения")
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "numpy_lab/data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива

    Example:
    >>> data = load_dataset("numpy_lab/data/students_scores.csv")
    >>> data.shape
    (10, 3)
    """
    # Подсказка: используйте pd.read_csv(path).to_numpy()

    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict[str, float]:
    """
    Представьте, что данные — это результаты экзамена по математике.
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    https://numpy.org/doc/stable/reference/generated/numpy.median.html
    https://numpy.org/doc/stable/reference/generated/numpy.std.html
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    
    Example:
    >>> data = np.array([10, 20, 30, 40, 50])
    >>> stats = statistical_analysis(data)
    >>> stats
    {'mean': 30.0, 'median': 30.0, 'std': 14.142135623730951, 'min': 10.0, 'max': 50.0, 'percentile_25': 20.0, 'percentile_75': 40.0}
    >>> stats['mean']
        30.0
    """
    # Подсказка: используйте np.mean(), np.median(), np.std(), 
    # np.min(), np.max(), np.percentile(data, 25), np.percentile(data, 75)

    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentile_25": float(np.percentile(data, 25)),
        "percentile_75": float(np.percentile(data, 75))
    }


def normalize_data(data) -> np.ndarray:
    """
    Min-Max нормализация.
    
    Формула: (x - min) / (max - min)
    
    Args:
        data (numpy.ndarray): Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    # Подсказка: вычислите min и max с помощью np.min() и np.max()
    pass

    min_val = np.min(data)
    max_val = np.max(data)

    if max == min:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения оценок по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    # Подсказка: используйте plt.hist(), добавьте заголовок, подписи осей,
    # сохраните в папку plots с помощью plt.savefig()

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, color='blue', edgecolor='black') #разбивает данные на 10 столбцов
    
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Количество студентов")

    plt.savefig('plots/math_histogram.png')
    plt.close() 


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции предметов.

    Изучить:
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    # Подсказка: используйте sns.heatmap(), добавьте заголовок, сохраните

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, 
                annot=True, 
                cmap='coolwarm', 
                xticklabels=['Математика', 'Физика', 'Информатика'],
                yticklabels=['Математика', 'Физика', 'Информатика'])
    
    plt.title('Корреляционная матрица оценок по предметам')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    
    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    # Подсказка: используйте plt.plot(), добавьте заголовок, подписи осей,
    # сохраните график

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, linewidth=2, markersize=8) #markersize - размер точек
    plt.title('Оценки студентов по математике')
    plt.xlabel('Номер студента')
    plt.ylabel('Оценка')
    plt.grid(True, alpha=0.3) # полупрозрачная сетка на фоне
    plt.xticks(x)  # отображение каждого значения(номера студента) на оси X 
    plt.savefig('plots/student_scores_line.png')
    plt.close()


# ============================================================
# ========================== ТЕСТЫ ===========================
# ============================================================

def test_create_vector():
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    assert np.array_equal(
        vector_add(np.array([1,2,3]), np.array([4,5,6])),
        np.array([5,7,9])
    )
    assert np.array_equal(
        vector_add(np.array([0,1]), np.array([1,1])),
        np.array([1,2])
    )


def test_scalar_multiply():
    assert np.array_equal(
        scalar_multiply(np.array([1,2,3]), 2),
        np.array([2,4,6])
    )


def test_elementwise_multiply():
    assert np.array_equal(
        elementwise_multiply(np.array([1,2,3]), np.array([4,5,6])),
        np.array([4,10,18])
    )


def test_dot_product():
    assert dot_product(np.array([1,2,3]), np.array([4,5,6])) == 32
    assert dot_product(np.array([2,0]), np.array([3,5])) == 6


def test_matrix_multiply():
    A = np.array([[1,2],[3,4]])
    B = np.array([[2,0],[1,2]])
    assert np.array_equal(matrix_multiply(A,B), A @ B)


def test_matrix_determinant():
    A = np.array([[1,2],[3,4]])
    assert round(matrix_determinant(A),5) == -2.0


def test_matrix_inverse():
    A = np.array([[1,2],[3,4]])
    invA = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system():
    A = np.array([[2,1],[1,3]])
    b = np.array([1,2])
    x = solve_linear_system(A,b)
    assert np.allclose(A @ x, b)


def test_load_dataset():
    # Для теста создадим временный файл
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78,81,90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([10,20,30])
    result = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    data = np.array([0,5,10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0,0.5,1]))


def test_plot_histogram():
    # Просто проверяем, что функция не падает
    data = np.array([1,2,3,4,5])
    plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1,0.5],[0.5,1]])
    plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    plot_line(x, y)


if __name__ == "__main__":
    print("Запустите python3 -m pytest test.py -v для проверки лабораторной работы.")
    # pytest

# .\numpy_env\Scripts\Activate.ps1