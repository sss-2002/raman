# arrangement/utils/file_handler.py
import numpy as np
import zipfile
import re

class FileHandler:
    def load_data_from_zip(self, zip_file):
        """从压缩包加载波数和光谱数据"""
        with zipfile.ZipFile(zip_file, 'r') as zf:
            file_list = zf.namelist()
            wavenumber_files = [f for f in file_list if 'wave' in f.lower() or 'wn' in f.lower() or '波数' in f]
            data_files = [f for f in file_list if 'spec' in f.lower() or 'data' in f.lower() or '光谱' in f]
            
            if not wavenumber_files:
                raise ValueError("未找到波数文件（含'wave'、'wn'或'波数'）")
            if not data_files:
                raise ValueError("未找到光谱数据文件（含'spec'、'data'或'光谱'）")

            wn_file = wavenumber_files[0]
            data_file = data_files[0]

            with zf.open(wn_file) as f:
                wavenumbers = np.loadtxt(f).ravel()

            with zf.open(data_file) as f:
                content = f.read().decode("utf-8")
                data = self._parse_data(content)

            return wavenumbers, data.T

    def _parse_data(self, content):
        """解析光谱数据内容"""
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        lines_list = content.splitlines()
        all_numbers = []
        for line in lines_list:
            all_numbers.extend(numb.findall(line))
        data = np.array([float(num) for num in all_numbers])
        n_rows = len(lines_list)
        n_cols = len(data) // n_rows if n_rows > 0 else 0
        
        if n_cols * n_rows != len(data):
            n_cols = len(data) // n_rows + 1
            data = data[:n_rows * n_cols]
        return data.reshape(n_rows, n_cols)

    def export_data(self, filename, data):
        with open(filename, "w") as f:
            for line in data.T:
                f.write("\t".join(map(str, line)) + "\n")
