import pydicom

# 指定DICOM文件路径
dicom_file_path = r"E:\乳腺数据集\至2023-7-12CESM图像\3-15图像\UCR202204210055\1.2.840.113619.2.255.22424451157206.22797220421083030.821.dcm"

# 读取DICOM文件
dicom_data = pydicom.dcmread(dicom_file_path)

# 打印DICOM文件的基本信息
sequence_number = dicom_data.get('(0020,0011)', None)
print(sequence_number)
print("Patient Name:", dicom_data.PatientName)
print("Study Date:", dicom_data.StudyDate)
print("Modality:", dicom_data.Modality)
print("Rows:", dicom_data.Rows)
print("Columns:", dicom_data.Columns)
print("Sequence Number:", dicom_data)

# 打印更多的元数据信息
print("\nDICOM Metadata:")
# for elem in dicom_data:
#     print(elem.tag, elem.description(), elem.value)

# 如果DICOM文件包含图像，您可以显示图像
import matplotlib.pyplot as plt
plt.imshow(dicom_data.pixel_array, cmap=plt.cm.bone)
plt.show()
