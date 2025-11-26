#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二章代码示例运行脚本
====================

此脚本用于依次运行第二章的所有代码示例，方便快速验证和学习。
"""

import subprocess
import sys
import os

def run_python_script(script_name):
    """运行Python脚本"""
    try:
        print(f"\n{'='*60}")
        print(f"正在运行 {script_name}...")
        print(f"{'='*60}")
        
        # 获取脚本路径
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=3600)
        
        # 输出结果
        if result.returncode == 0:
            print(f"✓ {script_name} 运行成功!")
            # 只显示前1000个字符以避免输出过长
            if len(result.stdout) > 1000:
                print(result.stdout[:1000] + "\n... (输出已截断)")
            else:
                print(result.stdout)
        else:
            print(f"✗ {script_name} 运行失败!")
            print("错误信息:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} 运行超时!")
    except Exception as e:
        print(f"✗ 运行 {script_name} 时发生错误: {str(e)}")

def main():
    """主函数"""
    print("第二章 深度学习入门 - 代码示例运行")
    print("=" * 60)
    
    # 定义要运行的脚本列表
    scripts = [
        "pytorch_tensors.py",
        "neural_network_basics.py",
        "car_mpg_predict_project.py",
        "iris_analysis_project.py",
        "mnist_fully_connected.py",
    ]

    # 运行每个脚本
    for script in scripts:
        run_python_script(script)

    print(f"\n{'='*60}")
    print("所有代码示例运行完成!")
    print("请查看各脚本生成的输出文件和图表。")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()