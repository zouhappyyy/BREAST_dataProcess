import os
import os.path as osp
from typing import Any, Optional, Dict, List
import pydicom
import pandas as pd
import tqdm
import argparse


# 提取Dicom相关信息
def extract_dicom_info(dicom_file_path: str) -> Optional[Dict[str, Any]]:
    # 读取DICOM文件
    dicom_data: pydicom.dataset.FileDataset = pydicom.dcmread(dicom_file_path)

    # 如果模态不是钼靶，则丢弃
    if dicom_data.Modality != 'MG':
        print(f'Exclude: Modality except MG but {dicom_data.Modality} -> {dicom_file_path}')
        return None

    try:
        # DICOM文件的基本信息
        extracted_info: Dict[str, Any] = {
            '病人标识符': dicom_data.PatientID,
            '序列号': dicom_data.SeriesNumber,
            '实例号': dicom_data.InstanceNumber,
            '成像类型': dicom_data.ImageType,
            '图像偏侧性': dicom_data.ImageLaterality,
            '投照体位': dicom_data.ViewPosition,
            '采集设备处理代码': dicom_data['AcquisitionDeviceProcessingCode'].value,  # Acquisition Device Processing Code
            '检查部位': dicom_data.BodyPartExamined,
            '模态': dicom_data.Modality,
        }
    # 丢弃故障数据
    except KeyError as e:
        print(f'Exclude: Key missing -> {dicom_file_path}')
        # print(dicom_data)
        return None

    # 如果DICOM文件包含图像，您可以显示图像
    # import matplotlib.pyplot as plt
    # plt.imshow(dicom_data.pixel_array, cmap=plt.cm.bone)
    # plt.show()

    return extracted_info


# 合成样本名称
def merge_sample_name(extracted_info: Dict[str, Any]) -> str:
    acquisition_device_processing_code: str = extracted_info['采集设备处理代码']
    short_processing_code: str = str.split(acquisition_device_processing_code, '_')[1]
    sample_name: str = '_'.join([
        extracted_info['病人标识符'],
        extracted_info['投照体位'],
        extracted_info['图像偏侧性'],
        short_processing_code,
    ])
    return sample_name


# 扫描数据目录，生成数据源索引表
def generate_data_source_indexer(source_root_path: str) -> List[Dict]:
    indexer: List[Dict] = []
    tot_count: int = 0
    with tqdm.tqdm(desc='Progress') as pbar:
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
                accession_number: str = osp.split(root)[-1]
                item: Dict[str, Any] = {
                    'accession_number': accession_number,
                    'dcm_file_name': dcmf,
                    'dcm_rel_path': osp.relpath(dcm_abs_path, source_root_path),
                    'target_rel_path': osp.join(accession_number, merge_sample_name(extracted_info))
                }
                indexer.append(item)
                pbar.update(1)
                tot_count += 1

    print(f'Total: {tot_count}, Valid: {len(indexer)}, Excluded: {tot_count - len(indexer)}')
    return indexer


# 数据源索引表转换为组表
def indexer_to_dataframe(indexer: List[Dict]) -> pd.DataFrame:
    if indexer is None or len(indexer) == 0: return None
    data_dict: Dict = {k: [v[k] for v in indexer] for k in indexer[0].keys()}
    return pd.DataFrame(data_dict)


def main(args):
    pass

if __name__ == '__main__':
    # extracted_info: Dict[str, Any] = extract_dicom_info(
    #     r'F:\CBIBF3\storage\Dataset\NII_Mammography\至2023-7-12CESM图像\3-15图像\UCR202204210055\1.2.840.113619.2.255.22424451157206.22797220421082853.811.dcm')
    # print(f'extracted_info: {extracted_info}')
    # sample_name: str = merge_sample_name(extracted_info)
    # print(f'sample_name: {sample_name}')
    # indexer: List[Dict] = generate_data_source_indexer(r'E:\乳腺数据集\至2023-7-12CESM图像\3-15图像')
    # indexer_dataframe: pd.DataFrame = indexer_to_dataframe(indexer)
    # indexer_dataframe.to_csv('generated_materials\data_source_indexer\indexer.csv', index=False)

    # dicom_data: pydicom.dataset.FileDataset = pydicom.dcmread(r"E:\乳腺数据集\至2023-7-12CESM图像\3-15图像\UCR202204210055\1.2.840.113619.2.255.22424451157206.22797220421083112.831.dcm")
    # print(dicom_data.dir())
    # for k in dicom_data.dir():
    #     print(dicom_data[k])
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process data in a specified root path.")

    # 添加 rootpath 参数
    parser.add_argument('--rootpath', type=str, help='Specify the root path for data processing', required=True)
    parser.add_argument('--csvpath', type=str, help='Specify the root path for data processing', required=True)

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    root_path = args.rootpath
    csv_path = args.csvpath

    indexer: List[Dict] = generate_data_source_indexer(root_path)
    indexer_dataframe: pd.DataFrame = indexer_to_dataframe(indexer)
    indexer_dataframe.to_csv(csv_path, index=False)
