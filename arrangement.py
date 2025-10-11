import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np

# 解压上传的文件
def extract_zip(uploaded_file, extract_to):
    """
    解压上传的zip文件到指定目录
    """
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

# 处理文件内容，假设数据是CSV格式
def load_data_from_zip(extract_to):
    """
    从解压的文件夹中读取数据文件，假设文件格式为CSV
    """
    # 获取解压后的文件列表
    files = os.listdir(extract_to)
    data_files = [f for f in files if f.endswith('.csv')]  # 假设数据是CSV文件
    
    if len(data_files) == 0:
        raise FileNotFoundError("没有找到CSV数据文件。")
    
    data_file = data_files[0]  # 取第一个CSV文件作为数据文件
    
    # 加载数据
    data_path = os.path.join(extract_to, data_file)
    data = pd.read_csv(data_path)
    return data

def main():
    # 设置页面
    st.set_page_config(page_title="上传光谱数据", layout="wide")

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

        # 读取并处理数据
        try:
            data = load_data_from_zip(extract_dir)
            st.write("光谱数据：")
            st.dataframe(data.head())  # 显示数据的前几行
        except Exception as e:
            st.error(f"加载数据失败: {str(e)}")
        
        # 标签输入
        st.subheader("样本标签")
        num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
        labels_input = st.text_input(
            "标签（逗号分隔，与光谱顺序一致）", 
            placeholder="例：0,0,1,1",
            key="labels_in"
        )
        
        if labels_input:
            try:
                labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                st.session_state.labels = labels
                st.success("标签已成功输入！")
            except Exception as e:
                st.warning(f"标签格式错误: {str(e)}")
        else:
            st.warning("⚠️ 请提供样本标签。")

        # 训练测试划分
        st.subheader("训练测试划分")
        train_test_ratio = st.slider(
           "训练集比例",
           min_value=0.1,
           max_value=0.9,
           value=0.8,
           step=0.1,
           format="%.1f",
           key="train_ratio"
        )

        # 按照比例划分数据
        if 'labels' in st.session_state:
            labels = st.session_state.labels
            n_samples = len(labels)
            train_size = int(n_samples * train_test_ratio)
            indices = np.random.permutation(n_samples)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            st.session_state.train_indices = train_indices
            st.session_state.test_indices = test_indices

            st.success(f"训练集: {len(train_indices)} 样本， 测试集: {len(test_indices)} 样本")
        else:
            st.warning("⚠️ 请提供样本标签，以便进行训练测试划分。")

if __name__ == "__main__":
    main()
