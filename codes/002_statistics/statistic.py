import os
import os.path as osp
from typing import Any, Optional, Dict, List
import itk
import pandas as pd
import numpy as np
import tqdm
import argparse


def get_nii_dict(nii_file_path: str) -> Dict[str, Any]:
    nii = itk.imread(nii_file_path)
    nii_dict = itk.dict_from_image(nii)
    data = nii_dict['data'].squeeze(axis=0)
    nii_dict['size'] = data.shape  # HxW
    nii_dict['data'] = data
    nii_dict['range'] = (np.min(data), np.max(data))
    nii_dict['mean'] = np.mean(data)
    nii_dict['std'] = np.std(data)
    return nii_dict


modes: List[str] = ['CC_L_FFDM', 'CC_R_FFDM', 'CC_L_CESM', 'CC_R_CESM', 'MLO_L_FFDM', 'MLO_R_FFDM', 'MLO_L_CESM', 'MLO_R_CESM']
vols: List[str] = ['HEIGHT', 'WIDTH', 'MIN', 'MAX', 'MEAN', 'STD']


# 扫描数据目录，生成数据源索引表
def generate_data_source_indexer(source_root_path: str) -> List[Dict[str, Any]]:
    indexer: List[Dict[str, Any]] = []
    mode_vols: List[str] = [mv for m in modes for mv in [m] + [f'{m}_{v}' for v in vols]]
    with tqdm.tqdm(desc='Indexing') as pbar:
        for root, dirs, files in os.walk(source_root_path):
            if len(files) == 0:
                continue
            item: Dict[str, Any] = {'登记号': osp.basename(root)}
            item.update({mv: None for mv in mode_vols})
            for m in modes:
                item[m] = len([0 for f in files if f.find(m) != -1])
                # 模式存在，提取统计信息
                if item[m] > 0:
                    nii_dict: Dict[str, Any] = get_nii_dict(osp.join(root, f'{item["登记号"]}_{m}.nii.gz'))
                    item[f'{m}_HEIGHT'] = nii_dict['size'][0]
                    item[f'{m}_WIDTH'] = nii_dict['size'][1]
                    item[f'{m}_MIN'] = nii_dict['range'][0]
                    item[f'{m}_MAX'] = nii_dict['range'][1]
                    item[f'{m}_MEAN'] = nii_dict['mean']
                    item[f'{m}_STD'] = nii_dict['std']

            indexer.append(item)
            pbar.update(1)
    return indexer


# 数据源索引表转换为组表
def indexer_to_dataframe(indexer: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if indexer is None or len(indexer) == 0:
        return None
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
