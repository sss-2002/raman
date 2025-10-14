光谱分析预处理工具
这是一个用于光谱数据分析的预处理工具，支持多种预处理算法和可视化功能。

功能特点
支持多种光谱预处理算法（基线校正、滤波、缩放、挤压等）
提供算法排列组合功能，可尝试不同预处理顺序
集成KNN分类器进行模型评估
丰富的可视化功能（原始光谱、预处理后光谱、k值曲线、混淆矩阵）
支持数据导入导出
安装方法
克隆仓库git clone https://github.com/yourusername/spectral-analysis-app.git cd spectral-analysis-app
安装依赖pip install -r requirements.txt
运行应用streamlit run main.py
使用指南
上传包含波数和光谱数据的压缩包
设置样本标签和训练测试比例
选择预处理方法和参数
点击"显示排列"生成不同的预处理方案组合
选择合适的方案并应用
设置k值并进行分类测试
查看结果并导出处理后的数据
项目结构
main.py - 主应用程序入口
algorithms/ - 预处理和分类算法实现
preprocessing.py - 预处理算法
classification.py - 分类算法
utils/ - 工具类
file_handler.py - 文件处理功能
visualization.py - 可视化功能
