import os
import os.path as osp
from typing import Any, Optional, Dict, List
import pandas as pd
import tqdm
import argparse


# 扫描数据目录，生成数据源索引表
def generate_data_source_indexer(source_root_path: str) -> List[Dict[str, Any]]:
    indexer: List[Dict[str, Any]] = []
    with tqdm.tqdm(desc='Indexing') as pbar:
        for root, dirs, files in os.walk(source_root_path):
            if len(files) == 0: continue
            item: Dict[str, Any] = {
                '登记号': osp.basename(root),
                'CC_L_FFDM': None,
                'CC_R_FFDM': None,
                'CC_L_CESM': None,
                'CC_R_CESM': None,
                'MLO_L_FFDM': None,
                'MLO_R_FFDM': None,
                'MLO_L_CESM': None,
                'MLO_R_CESM': None
            }
            for k in item.keys():
                if k == '登记号': continue
                item[k] = len([0 for f in files if f.find(k) != -1])
            indexer.append(item)
            pbar.update(1)
    return indexer


# 数据源索引表转换为组表
def indexer_to_dataframe(indexer: List[Dict[str, Any]]) -> pd.DataFrame:
    if indexer is None or len(indexer) == 0: return None
    data_dict: Dict = {k: [v[k] for v in indexer] for k in indexer[0].keys()}
    return pd.DataFrame(data_dict)


def main(args: argparse.Namespace):
    indexer: List[Dict[str, Any]] = generate_data_source_indexer(args.data_source_path)
    df = indexer_to_dataframe(indexer)
    df.to_csv(args.manifest_csv_log_path, index=False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="NIFTI转换结果统计例程")

    # 参数定义
    parser.add_argument('-dsp', '--data_source_path', type=str, help='钼靶NIFTI数据源根目录', required=True)
    parser.add_argument('-mcp', '--manifest_csv_log_path', type=str, help='CSV清单日志文件输出路径', required=True)

    # python statistic.py --data_source_path 钼靶NIFTI数据源根目录 --manifest_csv_log_path 数据源清单文件

    # 解析命令行参数
    args: argparse.Namespace = parser.parse_args()
    print(args)

    main(args)
