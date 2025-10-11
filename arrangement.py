import streamlit as st
import zipfile
import os

def extract_zip(uploaded_file, extract_to):
    """
    解压上传的zip文件到指定目录
    """
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def main():
    # 设置页面
    st.set_page_config(page_title="上传压缩包示例", layout="wide")

    # 文件上传控件
    uploaded_file = st.file_uploader("上传压缩包（zip格式）", type=["zip"])

    if uploaded_file is not None:
        # 显示上传的文件名
        st.write(f"上传的文件名：{uploaded_file.name}")

        # 设置解压缩目录
        extract_dir = "extracted_files"

        # 解压文件
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        # 调用解压函数
        extract_dir = extract_zip(uploaded_file, extract_dir)

        st.success(f"文件已成功解压到: {extract_dir}")

        # 列出解压后的文件
        extracted_files = os.listdir(extract_dir)
        st.write("解压后的文件：")
        for file in extracted_files:
            st.write(file)

if __name__ == "__main__":
    main()

