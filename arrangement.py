import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import numpy as np

class PermutationPreprocessor:
    def __init__(self, root):
        self.root = root
        self.root.title("排列预处理工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TCheckbutton", font=("SimHei", 10))
        self.style.configure("TRadiobutton", font=("SimHei", 10))
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建输入区域
        self.create_input_section()
        
        # 创建预处理选项区域
        self.create_preprocessing_options()
        
        # 创建结果展示区域
        self.create_result_section()
        
        # 创建按钮区域
        self.create_button_section()
        
    def create_input_section(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="输入数据", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="请输入需要处理的序列（用逗号分隔）:").pack(anchor=tk.W, pady=5)
        
        self.input_entry = ttk.Entry(input_frame, width=80)
        self.input_entry.pack(anchor=tk.W, pady=5, fill=tk.X)
        self.input_entry.insert(0, "1,3,5,2,4,6,8,7,9")  # 默认示例数据
        
        ttk.Label(input_frame, text="或随机生成序列:").pack(anchor=tk.W, pady=5)
        
        random_frame = ttk.Frame(input_frame)
        random_frame.pack(anchor=tk.W, pady=5)
        
        ttk.Label(random_frame, text="元素数量:").pack(side=tk.LEFT, padx=5)
        self.random_count = tk.StringVar(value="10")
        ttk.Entry(random_frame, textvariable=self.random_count, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(random_frame, text="生成随机序列", command=self.generate_random_sequence).pack(side=tk.LEFT, padx=5)
        
    def create_preprocessing_options(self):
        options_frame = ttk.LabelFrame(self.main_frame, text="预处理选项", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        
        # 预处理类型选择
        ttk.Label(options_frame, text="选择预处理类型:").pack(anchor=tk.W, pady=5)
        
        self.preprocess_type = tk.StringVar(value="basic")
        
        ttk.Radiobutton(options_frame, text="基础排序", variable=self.preprocess_type, value="basic").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="去重后排序", variable=self.preprocess_type, value="unique").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="反向排序", variable=self.preprocess_type, value="reverse").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="打乱顺序", variable=self.preprocess_type, value="shuffle").pack(anchor=tk.W)
        
        # 额外选项
        self.sort_ascending = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="升序排列（基础排序时生效）", variable=self.sort_ascending).pack(anchor=tk.W, pady=5)
        
    def create_result_section(self):
        result_frame = ttk.LabelFrame(self.main_frame, text="处理结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(result_frame, text="原始序列:").pack(anchor=tk.W)
        self.original_text = scrolledtext.ScrolledText(result_frame, height=3, wrap=tk.WORD)
        self.original_text.pack(fill=tk.X, pady=5)
        self.original_text.config(state=tk.DISABLED)
        
        ttk.Label(result_frame, text="处理后序列:").pack(anchor=tk.W)
        self.processed_text = scrolledtext.ScrolledText(result_frame, height=3, wrap=tk.WORD)
        self.processed_text.pack(fill=tk.X, pady=5)
        self.processed_text.config(state=tk.DISABLED)
        
        ttk.Label(result_frame, text="处理日志:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(result_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
    def create_button_section(self):
        button_frame = ttk.Frame(self.main_frame, padding="10")
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="执行预处理", command=self.process_sequence).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="清除结果", command=self.clear_results).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=10)
        
    def generate_random_sequence(self):
        try:
            count = int(self.random_count.get())
            if count <= 0:
                messagebox.showerror("错误", "请输入有效的元素数量")
                return
                
            sequence = [random.randint(1, 100) for _ in range(count)]
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, ",".join(map(str, sequence)))
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def process_sequence(self):
        # 清除之前的结果
        self.clear_results()
        
        # 获取输入序列
        input_str = self.input_entry.get().strip()
        if not input_str:
            messagebox.showerror("错误", "请输入序列或生成随机序列")
            return
            
        try:
            # 解析输入序列
            sequence = list(map(int, input_str.split(',')))
            self.log(f"解析输入序列: {sequence}")
            
            # 显示原始序列
            self.original_text.config(state=tk.NORMAL)
            self.original_text.insert(tk.END, ", ".join(map(str, sequence)))
            self.original_text.config(state=tk.DISABLED)
            
            # 根据选择的预处理类型进行处理
            processed = sequence.copy()
            preprocess_type = self.preprocess_type.get()
            
            if preprocess_type == "basic":
                self.log("执行基础排序...")
                processed.sort(reverse=not self.sort_ascending.get())
                order = "升序" if self.sort_ascending.get() else "降序"
                self.log(f"完成{order}排序")
                
            elif preprocess_type == "unique":
                self.log("执行去重后排序...")
                processed = list(np.unique(processed))
                if self.sort_ascending.get():
                    processed.sort()
                else:
                    processed.sort(reverse=True)
                order = "升序" if self.sort_ascending.get() else "降序"
                self.log(f"完成去重并按{order}排序")
                
            elif preprocess_type == "reverse":
                self.log("执行反向排序...")
                processed = processed[::-1]
                self.log("完成反向排序")
                
            elif preprocess_type == "shuffle":
                self.log("执行打乱顺序...")
                random.shuffle(processed)
                self.log("完成打乱顺序")
            
            # 显示处理后的序列
            self.processed_text.config(state=tk.NORMAL)
            self.processed_text.insert(tk.END, ", ".join(map(str, processed)))
            self.processed_text.config(state=tk.DISABLED)
            
            self.log("预处理完成!")
            
        except ValueError:
            messagebox.showerror("错误", "输入格式不正确，请使用逗号分隔的数字")
        except Exception as e:
            messagebox.showerror("错误", f"处理过程中发生错误: {str(e)}")
            
    def clear_results(self):
        self.original_text.config(state=tk.NORMAL)
        self.original_text.delete(1.0, tk.END)
        self.original_text.config(state=tk.DISABLED)
        
        self.processed_text.config(state=tk.NORMAL)
        self.processed_text.delete(1.0, tk.END)
        self.processed_text.config(state=tk.DISABLED)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    # 确保中文显示正常
    app = PermutationPreprocessor(root)
    root.mainloop()
