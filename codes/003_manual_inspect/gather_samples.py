import os
import os.path as osp
import shutil
from typing import Any, Optional, Dict, List
import pandas as pd
import tqdm
import argparse


def main(args: argparse.Namespace):
    src_path: str = args.data_source_path
    target_path: str = args.target_path
    os.makedirs(target_path, exist_ok=True)
    csv_path: str = args.inspect_csv_path
    df: pd.DataFrame = pd.read_csv(csv_path)
    gather_mode = args.gather_mode
    gather_func = lambda src, dst: shutil.copy(src, dst)
    if gather_mode == 'remove':
        gather_func = lambda src, dst: os.remove(src)
    elif gather_mode == 'move':
        gather_func = lambda src, dst: shutil.move(src, dst)

    with tqdm.tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():
            if '可用性意见' in row and row['可用性意见'] > 1:
                pbar.set_postfix(ignore=True)
                pbar.update(1)
                continue
            accession: str = row['登记号']
            mode: str = row['模式']
            src: str = osp.join(src_path, accession, f'{accession}_{mode}.nii.gz')
            dst: str = osp.join(target_path, osp.basename(src))
            gather_func(src, dst)
            pbar.set_postfix(**{gather_mode: osp.basename(src)})
            pbar.update(1)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="NIFTI样本提取例程")

    # 参数定义
    parser.add_argument('-dsp', '--data_source_path', type=str, help='钼靶NIFTI数据源根目录', required=True)
    parser.add_argument('-tp', '--target_path', type=str, help='提取样本目标目录', default='')
    parser.add_argument('-isp', '--inspect_csv_path', type=str, help='CSV人工检视样本清单', required=True)
    parser.add_argument('-m', '--gather_mode', type=str, help='提取方式[删除,移动,拷贝]', choices=['remove', 'move', 'copy'], default='copy')

    # python gather_samples.py --data_source_path 钼靶NIFTI数据源根目录 --target_path 提取样本目标目录 --inspect_csv_path CSV人工检视样本清单

    # 解析命令行参数
    args: argparse.Namespace = parser.parse_args()
    print(args)

    main(args)
