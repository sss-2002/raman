import numpy as np
import re
import zipfile
import os

class FileHandler:
    def load_data_from_zip(self, zip_file):
        """从压缩包中加载波数和光谱数据，自动识别数据维度"""
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # 列出压缩包中的所有文件
            file_list = zf.namelist()

            # 尝试识别波数文件和光谱数据文件
            wavenumber_files = [f for f in file_list if 'wave' in f.lower() or 'wn' in f.lower() or '波数' in f]
            data_files = [f for f in file_list if 'spec' in f.lower() or 'data' in f.lower() or '光谱' in f]

            if not wavenumber_files:
                raise ValueError("压缩包中未找到波数文件（通常包含'wave'、'wn'或'波数'）")
            if not data_files:
                raise ValueError("压缩包中未找到光谱数据文件（通常包含'spec'、'data'或'光谱'）")

            # 取第一个符合条件的文件
            wn_file = wavenumber_files[0]
            data_file = data_files[0]

            # 读取波数文件
            with zf.open(wn_file) as f:
                wavenumbers = np.loadtxt(f).ravel()

            # 读取光谱数据文件
            with zf.open(data_file) as f:
                content = f.read().decode("utf-8")
                data = self._parse_data(content)

            return wavenumbers, data.T

    def _parse_data(self, content):
        """解析光谱数据内容，自动识别数据维度"""
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        lines_list = content.splitlines()

        # 提取所有数字
        all_numbers = []
        for line in lines_list:
            all_numbers.extend(numb.findall(line))

        # 尝试确定数据形状
        # 假设波数长度为数据点数
        # 光谱条数 = 总数据点 / 数据点数
        # 这里先简单处理为二维数组
        data = np.array([float(num) for num in all_numbers])

        # 尝试合理的形状（假设每行数据点大致相等）
        # 先按行数划分
        n_rows = len(lines_list)
        n_cols = len(data) // n_rows if n_rows > 0 else 0

        if n_cols * n_rows != len(data):
            # 如果无法完美划分，调整最后一行
            n_cols = len(data) // n_rows + 1
            data = data[:n_rows * n_cols]  # 截断多余数据

        return data.reshape(n_rows, n_cols)

    def export_data(self, filename, data):
        with open(filename, "w") as f:
            for line in data.T:  # 转置回原始格式
                f.write("\t".join(map(str, line)) + "\n")
