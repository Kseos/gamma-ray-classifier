# src/utils.py

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def scale_and_oversample(df, oversample=False):
    """
    Масштабирует признаки и применяет RandomOverSampler для балансировки классов.

    Parameters
    ----------
    df : pd.DataFrame
        Датасет с признаками и целевой переменной (class). Целевая переменная должна быть последней колонкой.
    oversample : bool, optional
        Если True, применяется RandomOverSampler для балансировки классов. По умолчанию False.

    Returns
    -------
    tuple
        - data (ndarray): Масштабированные данные с добавленной целевой переменной.
        - X_scaled (ndarray): Масштабированные признаки.
        - y (ndarray): Целевая переменная.
    """

    X = df[df.columns[:-1]].values  # Все колонки кроме последней
    y = df[df.columns[-1]].values  # Последняя колонка - это целевая переменная (class)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        # Применяем овер-сэмплинг
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y
