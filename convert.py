import os
import os.path as osp
from typing import Any, Optional, Dict, List
import pydicom
import itk
import pandas as pd
import tqdm
import argparse


# 提取Dicom相关信息
def extract_dicom_info(dicom_file_path: str) -> Optional[Dict[str, Any]]:
    # 读取DICOM文件
    dicom_data: pydicom.dataset.FileDataset = pydicom.dcmread(dicom_file_path)

    # 如果模态不是钼靶，则丢弃
    if dicom_data.Modality != 'MG':
        print(f'Exclude: Modality expect MG but {dicom_data.Modality} -> {dicom_file_path}')
        return None

    try:
        # DICOM文件的基本信息
        extracted_info: Dict[str, Any] = {
            '登记号': dicom_data.AccessionNumber,
            '病人标识符': dicom_data.PatientID,
            '序列号': dicom_data.SeriesNumber,
            '实例号': dicom_data.InstanceNumber,
            '字符集': dicom_data.SpecificCharacterSet,
            '生成日期': dicom_data.ContentDate,
            '生成时间': dicom_data.ContentTime,
            '成像类型': dicom_data.ImageType,
            '图像偏侧性': dicom_data.ImageLaterality,
            '投照体位': dicom_data.ViewPosition,
            '临床视角': dicom_data[0x0045, 0x101b].value if (0x0045, 0x101b) in dicom_data else 'UND',  # Clinical View
            '采集设备处理代码': dicom_data['AcquisitionDeviceProcessingCode'].value if 'AcquisitionDeviceProcessingCode' in dicom_data else 'UND',
            # [AUTOMATIC, MANUAL]  手动/自动曝光
            '曝光控制模式': dicom_data['ExposureControlMode'].value if 'ExposureControlMode' in dicom_data else 'UND',
            # [LIN, LOG] X射线强度到像素值的映射方式，常见的有线性和对数两种
            '像素明度相关性': dicom_data['PixelIntensityRelationship'].value if 'PixelIntensityRelationship' in dicom_data else 'UND',
            '检查部位': dicom_data.BodyPartExamined,
            '模态': dicom_data.Modality,
        }
    # 丢弃故障数据
    except KeyError as e:
        print(f'Exclude: Key \'{e}\' missing -> {dicom_file_path}')
        return None

    if extracted_info['临床视角'] != 'UND' and extracted_info['图像偏侧性'] + extracted_info['投照体位'] != extracted_info['临床视角']:
        print(
            f'Exclude: Device view {extracted_info["图像偏侧性"]}+{extracted_info["投照体位"]} not equal to Clinical view {extracted_info["临床视角"]}')
        return None

    return extracted_info


# 合成样本名称
def merge_sample_name(extracted_info: Dict[str, Any]) -> str:
    acquisition_device_processing_code: str = extracted_info['采集设备处理代码']
    short_processing_code: str = 'UND' if acquisition_device_processing_code == 'UND' else str.split(acquisition_device_processing_code, '_')[1]
    sample_name: str = '_'.join([
        extracted_info['登记号'],
        # extracted_info['病人标识符'],
        # extracted_info['曝光控制模式'],
        # extracted_info['像素明度相关性'],
        # extracted_info['生成日期'],
        # extracted_info['生成时间'],
        extracted_info['投照体位'],
        extracted_info['图像偏侧性'],
        short_processing_code,
    ])
    return sample_name


# 扫描数据目录，生成数据源索引表
def generate_data_source_indexer(source_root_path: str) -> List[Dict[str, Any]]:
    indexer: List[Dict[str, Any]] = []
    tot_count: int = 0
    convert_count: int = 0
    with tqdm.tqdm(desc='Indexing') as pbar:
        for root, dirs, files in os.walk(source_root_path):
            # print((root, dirs, files))
            # dcm_files: List[str] = [p for p in files if osp.splitext(p)[1] == '.dcm']
            dcm_files = files
            if len(dcm_files) == 0: continue
            for dcmf in dcm_files:
                dcm_abs_path: str = osp.abspath(osp.join(root, dcmf))
                extracted_info = extract_dicom_info(dcm_abs_path)
                if extracted_info is None:
                    pbar.update(1)
                    tot_count += 1
                    continue
                item: Dict[str, Any] = {
                    '登记号': extracted_info['登记号'],
                    'DICOM源文件名': dcmf,
                    'DICOM相对路径': osp.relpath(dcm_abs_path, source_root_path),
                    'Nifti相对路径': ''
                }
                if extracted_info['投照体位'] in {'CC', 'MLO'} and \
                        extracted_info['成像类型'][1] == 'PRIMARY' and \
                        extracted_info['成像类型'][3] in {'LOW_ENERGY', 'RECOMBINED'} and \
                        extracted_info['采集设备处理代码'] in {'GEMS_FFDM_PV', 'GEMS_CESM_1'} and \
                        extracted_info['像素明度相关性'] == 'LOG':
                    item['Nifti相对路径'] = osp.join(extracted_info['登记号'], merge_sample_name(extracted_info))
                    convert_count += 1
                item.update(extracted_info)
                indexer.append(item)
                pbar.update(1)
                tot_count += 1

    print(f'Total: {tot_count}, Convert: {convert_count}, Valid: {len(indexer)}, Excluded: {tot_count - len(indexer)}')
    return indexer


# 数据源索引表转换为组表
def indexer_to_dataframe(indexer: List[Dict[str, Any]]) -> pd.DataFrame:
    if indexer is None or len(indexer) == 0: return None
    data_dict: Dict = {k: [v[k] for v in indexer] for k in indexer[0].keys()}
    return pd.DataFrame(data_dict)


# 基于索引表将DICOM转换为NII，输出到指定路径
def dicom_convert_to_nii(indexer: List[Dict[str, Any]], source_root_path: str, target_root_path: str):
    """
    indexer: {
        '登记号': UCRxxxxxxxxxxxx作为目录,
        'DICOM源文件名': 源文件名称（带扩展名，不带目录）,
        'DICOM相对路径': 源文件相对于数据源根的相对路径,
        'Nifti相对路径': 输出到指定根相对路径
    }
    """
    convert_count: int = 0
    with tqdm.tqdm(total=len(indexer), desc='Converting') as pbar:
        for idx in indexer:
            if idx['Nifti相对路径'] == '':
                pbar.update(1)
                continue
            abs_source_file_path: str = osp.abspath(osp.join(source_root_path, idx['DICOM相对路径']))
            abs_target_file_path: str = osp.abspath(osp.join(target_root_path, idx['Nifti相对路径'] + '.nii.gz'))
            abs_target_dir_path: str = osp.dirname(abs_target_file_path)

            # 建立目录
            os.makedirs(abs_target_dir_path, exist_ok=True)

            # DICOM转换到NII
            # 读取DICOM文件
            image = itk.imread(abs_source_file_path)

            # 保存为Nitfi文件
            itk.imwrite(image, abs_target_file_path)
            pbar.update(1)
            convert_count += 1

    print(f'Total: {len(indexer)}, Convert: {convert_count}, Excluded: {len(indexer) - convert_count}')


def main(args: argparse.Namespace):
    # 获取参数值
    data_source_path = args.data_source_path
    manifest_csv_path = args.manifest_csv_log_path
    target_root_path = args.target_path

    indexer: List[Dict[str, Any]] = generate_data_source_indexer(data_source_path)

    # 打印数据源解析日志
    indexer_dataframe: pd.DataFrame = indexer_to_dataframe(indexer)
    indexer_dataframe.to_csv(manifest_csv_path, index=False)

    dicom_convert_to_nii(indexer, data_source_path, target_root_path)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="钼靶DICOM转NII例程")

    # 添加 rootpath 参数
    parser.add_argument('-dsp', '--data_source_path', type=str, help='钼靶DICOM数据源根目录', required=True)
    parser.add_argument('-mcp', '--manifest_csv_log_path', type=str, help='CSV清单日志文件输出路径', required=True)
    parser.add_argument('-tp', '--target_path', type=str, help='目标数据集根目录', required=True)

    # python convert.py --data_source_path 钼靶DICOM数据源根目录 --manifest_csv_log_path 数据源清单文件 --target_path 目标数据集根目录

    # 解析命令行参数
    args: argparse.Namespace = parser.parse_args()

    main(args)

    # 如果DICOM文件包含图像，您可以显示图像
    # import matplotlib.pyplot as plt
    # plt.imshow(dicom_data.pixel_array, cmap=plt.cm.bone)
    # plt.show()
