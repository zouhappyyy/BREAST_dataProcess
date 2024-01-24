import os
import os.path as osp
from glob import glob
from typing import Any, Optional, Dict, List, Tuple
import pydicom
import itk
import pandas as pd
import tqdm
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_nii_dict(nii_file_path: str) -> Dict[str, Any]:
    nii = itk.imread(nii_file_path)
    nii_dict = itk.dict_from_image(nii)
    data = nii_dict['data'].squeeze(axis=0)
    nii_dict['size'] = data.shape
    nii_dict['data'] = data
    nii_dict['range'] = (np.min(data), np.max(data))
    return nii_dict


def get_data_range(nii_file_path: str) -> Tuple[int, int]:
    nii = itk.imread(nii_file_path)
    data = itk.GetArrayFromImage(nii).squeeze(axis=0)
    return np.min(data), np.max(data)


def display_nii_image_with_metainfo(nii_file_path: str):
    nii = itk.imread(nii_file_path)
    print(f'dict: {itk.dict_from_image(nii)}')
    data = itk.GetArrayFromImage(nii).squeeze(axis=0)
    print(f'data_type: {type(data)}')
    print(f'size(HxW): {data.shape}')
    print(f'pixel range: ({np.min(data)}, {np.max(data)})')
    plt.title(f'Nifti Image: {osp.basename(nii_file_path)}')
    plt.imshow(data, cmap=plt.cm.bone)
    plt.show()


def display_histogram(nii_dict: Dict):
    pixel_range: Tuple[int, int] = nii_dict['range']
    hist_size = pixel_range[1] - pixel_range[0]
    data: np.ndarray = nii_dict['data']
    plt.title('Histogram')
    plt.hist(data.ravel(), range=pixel_range, bins=hist_size)
    plt.show()


RFunc = 'reduce_func'
DFunc = 'display_func'
SFunc = 'summary_func'
cur_path: str = ''


# 遍历样本统计
# funcs: {func_name: {reduce_func, summary_func, display_func}, ...}
def glob_samples(root_path: str, pattern: str, funcs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    global cur_path
    file_path: List[str] = glob(osp.join(root_path, pattern), recursive=True)
    func_tmp: Dict[str, Any] = {k: None for k in funcs.keys()}
    with tqdm.tqdm(total=len(file_path)) as pbar:
        for path in file_path:
            cur_path = path
            nii_dict: Dict[str, Any] = get_nii_dict(path)
            func_tmp = {k: f[RFunc](func_tmp[k], nii_dict) for k, f in funcs.items() if RFunc in f.keys()}
            display_dict: Dict[str, Any] = {k: f[DFunc](func_tmp[k] if k in func_tmp.keys() else None, nii_dict) for k, f in funcs.items() if
                                            DFunc in f.keys()}
            pbar.set_postfix(**display_dict, sample=osp.basename(path))
            pbar.update(1)
    func_summary: Dict[str, Any] = {k: funcs[k][SFunc](v) if SFunc in funcs[k].keys() else v for k, v in func_tmp.items() if v is not None}
    return func_summary


"""
统计函数
"""


# 打印当前图像值域
def current_min_max_display_func(reduced_after: Tuple[int, int], oprand: Dict[str, Any]):
    return oprand['range']


# 图像值域统计
def min_max_reduce_func(reduced: Optional[Tuple[int, int]], oprand: Dict[str, Any]) -> Tuple[int, int]:
    min_max: Tuple[int, int] = oprand['range']
    if reduced is None:
        return min_max
    return min(reduced[0], min_max[0]), max(reduced[1], min_max[1])


def min_max_display_func(reduced_after: Tuple[int, int], oprand: Dict[str, Any]) -> Tuple[int, int]:
    return reduced_after


def get_min_max_func_dict():
    return {RFunc: min_max_reduce_func, DFunc: min_max_display_func}


# 图像值分布统计
# reduced: [size, mean, var]
def mean_std_reduce_func(reduced: Optional[Tuple[int, float, float]], oprand: Dict[str, Any]) -> Tuple[int, float, float]:
    op_size, op_mean, op_var = oprand['data'].size, oprand['data'].mean(), oprand['data'].var()
    if reduced is None:
        return op_size, op_mean, op_var

    reduced_size, reduced_mean, reduced_var = reduced

    diff_mean: float = reduced_mean - op_mean
    prod_size: float = reduced_size * op_size

    combined_size: int = reduced_size + op_size
    combined_mean: float = (reduced_size * reduced_mean + op_size * op_mean) / combined_size
    combined_var: float = (reduced_size * reduced_var + op_size * op_var + (prod_size * diff_mean * diff_mean) / combined_size) / combined_size

    return combined_size, combined_mean, combined_var


def mean_std_display_func(reduced_after: Tuple[int, float, float], oprand: Dict[str, Any]) -> str:
    return f'({reduced_after[1]:.2f}, {np.sqrt(reduced_after[2]):.2f})'


def mean_std_summary_func(reduced_after: Tuple[int, float, float]) -> Tuple[int, float, float]:
    return reduced_after[0], reduced_after[1], np.sqrt(reduced_after[2])


def get_mean_std_func_dict():
    return {RFunc: mean_std_reduce_func, DFunc: mean_std_display_func, SFunc: mean_std_summary_func}


# [ROI值区间内] 图像值分布统计
# roi_domain: [min, max]
# reduced: [size, mean, var]
def roi_mean_std_reduce_func(
        roi_domain: Tuple[float, float],
        reduced: Optional[Tuple[int, float, float]],
        oprand: Dict[str, Any]) -> Tuple[int, float, float]:
    roi_data: np.ndarray = oprand['data'][np.bitwise_and(roi_domain[0] <= oprand['data'], oprand['data'] <= roi_domain[1])]
    if roi_data.size == 0:
        print(f'ROI Domain{roi_domain} not found: {cur_path}')
        return reduced

    op_size, op_mean, op_var = roi_data.size, roi_data.mean(), roi_data.var()
    if reduced is None:
        return op_size, op_mean, op_var

    reduced_size, reduced_mean, reduced_var = reduced

    diff_mean: float = reduced_mean - op_mean
    prod_size: float = reduced_size * op_size

    combined_size: int = reduced_size + op_size
    combined_mean: float = (reduced_size * reduced_mean + op_size * op_mean) / combined_size
    combined_var: float = (reduced_size * reduced_var + op_size * op_var + (prod_size * diff_mean * diff_mean) / combined_size) / combined_size

    return combined_size, combined_mean, combined_var


def get_roi_mean_std_func_dict(roi_domain: Tuple[float, float]):
    return {
        RFunc: lambda reduced, oprand: roi_mean_std_reduce_func(roi_domain, reduced, oprand),
        DFunc: mean_std_display_func,
        SFunc: mean_std_summary_func
    }


# 图像值计数
# 1D bincount
def bin_count_reduce_func(reduced: Optional[np.ndarray], oprand: Dict[str, Any]) -> np.ndarray:
    data = oprand['data']
    op_bin: np.ndarray = np.bincount(data.ravel())
    if reduced is None:
        return op_bin

    # 延拓较短的计数向量
    if reduced.size < op_bin.size:
        reduced = np.pad(reduced, (0, np.abs(reduced.size - op_bin.size)), 'constant', constant_values=0)
    elif reduced.size > op_bin.size:
        op_bin = np.pad(op_bin, (0, np.abs(reduced.size - op_bin.size)), 'constant', constant_values=0)

    return reduced + op_bin


def get_bin_count_func_dict():
    return {RFunc: bin_count_reduce_func}


"""
统计量：
    极值
    均值，标准差
    [ROI人体组织值区间内] 均值，标准差
    [ROI空腔值区间内] 均值，标准差
    [FFDM][ROI人体组织值区间内] 均值，标准差
    [FFDM][ROI空腔值区间内] 均值，标准差
    [CESM][ROI人体组织值区间内] 均值，标准差
    [CESM][ROI空腔值区间内] 均值，标准差
可视化：
    像素值直方图
    [ROI人体组织值区间内] 像素值直方图
    [ROI空腔值区间内] 像素值直方图
    [FFDM][ROI人体组织值区间内] 像素值直方图
    [FFDM][ROI空腔值区间内] 像素值直方图
    [CESM][ROI人体组织值区间内] 像素值直方图
    [CESM][ROI空腔值区间内] 像素值直方图
"""


# 提取各模式的统计信息
def glob_summary_for_patterns(root_path: str,
                              pattern: List[str],
                              tissue_domain: Tuple[float, float],
                              cavum_domain: Tuple[float, float]) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for p in pattern:
        summary = glob_samples(root_path,
                               pattern=r'**\*_' + p + '.nii.gz',
                               funcs={
                                   'min_max': get_min_max_func_dict(),
                                   'mean_std': get_mean_std_func_dict(),
                                   'tissue_mean_std': get_roi_mean_std_func_dict(tissue_domain),
                                   'cavum_mean_std': get_roi_mean_std_func_dict(cavum_domain),
                                   'bin_count': get_bin_count_func_dict()
                               })
        summaries[p] = summary
    return summaries


# 模式统计信息转换为组表
def summaries_to_dataframe(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if summaries is None or len(summaries) == 0: return None
    data_dict: Dict = {
        '模式': [],
        '值域下限': [],
        '值域上限': [],
        '均值': [],
        '标准差': [],
        'ROI人体组织均值': [],
        'ROI人体组织标准差': [],
        'ROI空腔均值': [],
        'ROI空腔标准差': []
    }
    for mode, summary in summaries.items():
        data_dict['模式'].append(mode)
        data_dict['值域下限'].append(summary['min_max'][0])
        data_dict['值域上限'].append(summary['min_max'][1])
        data_dict['均值'].append(summary['mean_std'][1])
        data_dict['标准差'].append(summary['mean_std'][2])
        data_dict['ROI人体组织均值'].append(summary['tissue_mean_std'][1])
        data_dict['ROI人体组织标准差'].append(summary['tissue_mean_std'][2])
        data_dict['ROI空腔均值'].append(summary['cavum_mean_std'][1])
        data_dict['ROI空腔标准差'].append(summary['cavum_mean_std'][2])
    return pd.DataFrame(data_dict)


# 值计数作图
def plot_value_histogram(summaries: Dict[str, Dict[str, Any]], plot_path: Optional[str]):
    if plot_path is not None:
        os.makedirs(plot_path, exist_ok=True)
    for mode, summary in summaries.items():
        display_mode: str = mode.replace('*', 'ALL')
        hist: np.ndarray = summary['bin_count']
        plt.bar(np.arange(len(hist)), hist, width=1.0)
        plt.title(f'{display_mode} Histogram')
        plt.xlabel('pixel value')
        plt.ylabel('count')
        plt.ylim(0, max(hist[1:]))
        if plot_path is not None:
            plt.savefig(osp.join(plot_path, f'{display_mode}_histogram.png'), dpi=300)
        plt.show()



def main(args):
    root_path: str = args.data_root_path
    pattern: List[str] = args.pattern
    tissue_domain: Tuple[float, float] = (args.tissue_domain[0], args.tissue_domain[1])
    cavum_domain: Tuple[float, float] = (args.cavum_domain[0], args.cavum_domain[1])

    summaries: Dict[str, Dict[str, Any]] = glob_summary_for_patterns(root_path, pattern, tissue_domain, cavum_domain)
    df: pd.DataFrame = summaries_to_dataframe(summaries)
    df.to_csv(args.analyze_csv_path, index=False)
    if args.analyze_plot_path is not None:
        plot_value_histogram(summaries, args.analyze_plot_path)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="样本模式统计信息提取例程")

    # 参数定义
    parser.add_argument('-drp', '--data_root_path', type=str, help='钼靶NIFTI数据源根目录', required=True)
    parser.add_argument('-acp', '--analyze_csv_path', type=str, help='统计数据输出文件路径', required=True)
    parser.add_argument('-app', '--analyze_plot_path', type=str, help='像素值计数统计图输出目录', default=None)
    parser.add_argument('-td', '--tissue_domain', type=float, nargs=2, default=[1500., 4100.], help='ROI人体组织值区间')
    parser.add_argument('-cd', '--cavum_domain', type=float, nargs=2, default=[0., 1500.], help='ROI空腔值区间')
    parser.add_argument('-p', '--pattern', type=str, nargs='+', default=['*_*_*'], help='模式列表，用_分隔，*为通配符')

    # python analyze.py --data_source_path 钼靶NIFTI数据源根目录 --analyze_csv_path 统计数据输出文件路径

    # 解析命令行参数
    args: argparse.Namespace = parser.parse_args()
    print(args)

    main(args)
