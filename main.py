#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实用人工智能教程 - 主入口点
=========================

这是一个用于运行各种人工智能实践示例的主程序。
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


def show_menu():
    """显示主菜单"""
    print("\n" + "="*50)
    print("实用人工智能教程")
    print("="*50)
    print("请选择要运行的章节:")
    print("0. 人工智能概述")
    print("1. Python 基础与数据处理")
    print("2. 机器学习基础")
    print("3. 深度学习入门")
    print("4. 计算机视觉")
    print("5. 自然语言处理")
    print("6. Transformer架构")
    print("q. 退出程序")
    print("-"*50)


def main():
    """主函数"""
    print("欢迎来到实用人工智能教程!")
    
    while True:
        show_menu()
        choice = input("请输入您的选择 (0-6, q): ").strip().lower()
        
        if choice == 'q':
            print("感谢使用本教程，再见!")
            break
        elif choice == '0':
            print("第 0 章: 人工智能概述")
            print("请查看 chapter0/README.md 文件了解详细内容")
            run_python_script('chapter0/code/run_all_examples.py')
        elif choice == '1':
            print("第 1 章: Python 基础与数据处理")
            print("请查看 chapter1/README.md 文件了解详细内容")
            run_python_script('chapter1/code/run_all_examples.py')
        elif choice == '2':
            print("第 2 章: 机器学习基础")
            print("请查看 chapter2/README.md 文件了解详细内容")
            run_python_script('chapter2/code/run_all_examples.py')
        elif choice == '3':
            print("第 3 章: 深度学习入门")
            print("请查看 chapter3/README.md 文件了解详细内容")
            run_python_script('chapter3/code/run_all_examples.py')
        elif choice == '4':
            print("第 4 章: 计算机视觉")
            print("请查看 chapter4/README.md 文件了解详细内容")
            run_python_script('chapter4/code/run_all_examples.py')
        elif choice == '5':
            print("第 5 章: 自然语言处理")
            print("请查看 chapter5/README.md 文件了解详细内容")
            run_python_script('chapter5/code/run_all_examples.py')
        elif choice == '6':
            print("第 6 章: Transformer架构")
            print("请查看 chapter6/README.md 文件了解详细内容")
            run_python_script('chapter6/code/run_all_examples.py')
        else:
            print("无效的选择，请重新输入!")


if __name__ == "__main__":
    main()