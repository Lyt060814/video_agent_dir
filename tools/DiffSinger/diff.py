import argparse
import os
import sys


def run_diffsinger(exp_name="0228_opencpop_ds100_rel", inp="", save_name=""):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run diff singer inference')
    parser.add_argument('--exp_name', default=exp_name, type=str, help='Experiment name')
    parser.add_argument('--inp', default=inp, type=str, help='audio info')
    parser.add_argument('--save_name', default=save_name, type=str, help='save name')
    args = parser.parse_args([])  # Empty list to avoid reading command line arguments

    # Get current working directory
    original_dir = os.getcwd()
    original_sys_path = sys.path.copy()  # 保存原始sys.path

    try:
        # Change to DiffSinger directory
        diffsinger_dir = os.path.join("tools", "DiffSinger")
        os.chdir(diffsinger_dir)

        # 添加DiffSinger目录到Python路径
        diffsinger_abs_path = os.path.abspath('.')
        sys.path.append(diffsinger_abs_path)

        # 或者直接修改PYTHONPATH环境变量
        os.environ["PYTHONPATH"] = diffsinger_abs_path

        # 使用subprocess运行
        import subprocess
        cmd_parts = [
            sys.executable,
            "inference/svs/ds_e2e.py",
            "--exp_name", args.exp_name,
            "--inp", args.inp,
            "--save_name", save_name
        ]

        process = subprocess.run(cmd_parts)
        return_code = process.returncode
        print(f"Process returned: {return_code}")
        return return_code

    finally:
        # 恢复原始环境
        os.chdir(original_dir)
        sys.path = original_sys_path  # 恢复原始sys.path