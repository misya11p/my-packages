from spleeter.separator import Separator
import numpy as np
from typing import Optional, Union


def separate(
    source: Union[str, np.ndarray],
    nstems: int,
    output_path: Optional[str] = None,
) -> Optional[dict]:
    """
    spleeterを使い易くした関数

    Args:
        source (Union[str, np.ndarray]):
            入力するファイルのパスまたはnumpy配列
        nstems (int):
            分離させるパートの数
        output_path (Optional[str]):
            出力先(ディレクトリ)のパス(入力がファイルパスの場合のみ)

    Returns:
        Optional[dir]: spleeterが返した辞書(入力がnumpy配列の場合のみ)
    """
    assert nstems not in (2, 4, 5), "nstems must be 2 or 4 or 5"

    separator = Separator(f'spleeter:{nstems}stems')

    if type(source) == str:
        assert not output_path, "output_path must not be None"
        separator.separate_to_file(source, output_path)

    else:
        if source.ndim == 1:
            source = source.reshape(-1, 1)
        elif source.shape[0] < source.shape[1]:
            source = source.T
        return separator.separate(source)