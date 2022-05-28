import numpy as np

def cos_similar(x: np.ndarray, y: np.ndarray) -> float:
    """
    コサイン類似度

    Args:
        x (np.ndarray): ベクトル
        y (np.ndarray): ベクトル

    Returns:
        float: コサイン類似度
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def softmax(x:np.ndarray) -> np.ndarray:
    """
    ソフトマックス関数

    Args:
        x (np.ndarray): 入力

    Returns:
        np.ndarray: 出力
    """
    return np.exp(x) / np.sum(np.exp(x), axis=-1)