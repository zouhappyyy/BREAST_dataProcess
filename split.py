import os
import os.path as osp
from typing import Any, Optional, Dict, List, Tuple
import pandas as pd
import tqdm
import shutil
from sklearn.model_selection import train_test_split
import argparse


# 扫描数据目录，生成数据源索引表
# 返回：(带有完整项目的登记号列表, 项目不全的登记号列表)
def load_csv_indexer(csv_path: str) -> Tuple[List[str], List[str]]:
    mods: List[str] = [
        'CC_L_FFDM',
        'CC_R_FFDM',
        'CC_L_CESM',
        'CC_R_CESM',
        'MLO_L_FFDM',
        'MLO_R_FFDM',
        'MLO_L_CESM',
        'MLO_R_CESM'
    ]
    df = pd.read_csv(csv_path)
    df['count'] = df[mods].sum(axis=1)
    complete_accessions: List[str] = df[df['count'] == len(mods)]['登记号'].to_list()
    defect_accessions: List[str] = df[df['count'] != len(mods)]['登记号'].to_list()
    return complete_accessions, defect_accessions


# 数据集的划分
# 返回：(训练集,验证集,测试集) 皆为登记号 Accession Number
def split_dataset(accession_index: List[str], split_ratio: List[float], random_seed: int) -> Tuple[List[str], List[str], List[str]]:
    tot: float = sum(split_ratio)
    train_ratio, validation_ratio, test_ratio = (r / tot for r in split_ratio)
    print(f'训练集:验证集:测试集={train_ratio}:{validation_ratio}:{test_ratio}')
    train_vt = train_test_split(accession_index, test_size=validation_ratio + test_ratio, random_state=random_seed)
    val_test = train_test_split(train_vt[1], test_size=test_ratio / (validation_ratio + test_ratio), random_state=random_seed)
    train_set = train_vt[0]
    validation_set = val_test[0]
    test_set = val_test[1]
    print(f'训练集:验证集:测试集={len(train_set)}:{len(validation_set)}:{len(test_set)}')
    return train_set, validation_set, test_set


# 拷贝目录
def copy_dir(src: str, dst: str):
    # print(f'move {src} -> {osp.join(dst, os.path.basename(src))}')
    shutil.move(src, osp.join(dst, os.path.basename(src)))


def copy_accession(source_root: str, accessions: List[str], dst: str):
    os.makedirs(dst, exist_ok=True)
    with tqdm.tqdm(total=len(accessions), desc=f'Move to {osp.basename(dst)}') as pbar:
        for ac in accessions:
            src: str = osp.join(source_root, ac)
            copy_dir(src, dst)
            pbar.update(1)


def main(args: argparse.Namespace):
    complete_accessions, defect_accessions = load_csv_indexer(args.statistic_csv)
    train_set, validation_set, test_set = split_dataset(complete_accessions, args.split_ratio, args.random_seed)
    copy_accession(args.data_source_path, defect_accessions, osp.join(args.data_source_path, 'defact_set'))
    copy_accession(args.data_source_path, train_set, osp.join(args.data_source_path, 'train_set'))
    copy_accession(args.data_source_path, validation_set, osp.join(args.data_source_path, 'validation_set'))
    copy_accession(args.data_source_path, test_set, osp.join(args.data_source_path, 'test_set'))


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="NIFTI结果划分例程")

    # 参数定义
    parser.add_argument('-s', '--random_seed', type=int, help='随机数种子', default=0)
    parser.add_argument('-dsp', '--data_source_path', type=str, help='钼靶NIFTI数据源根目录', required=True)
    parser.add_argument('-sc', '--statistic_csv', type=str, help='数据清单文件路径（由statistic.py例程生成）', required=True)
    parser.add_argument('-sr', '--split_ratio', type=float, nargs='+', default=[6.0, 3.0, 1.0], help='训练集:验证集:测试集')

    # python split.py --data_source_path 钼靶NIFTI数据源根目录

    # 解析命令行参数
    args: argparse.Namespace = parser.parse_args()
    print(args)

    main(args)
