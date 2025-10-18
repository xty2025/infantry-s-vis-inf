import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Parser with multiple types of arguments")

# 添加参数
parser.add_argument('-n', '--number', type=int, help="A number", required=True)
parser.add_argument('--float', type=float, help="A floating-point number", default=3.14)
parser.add_argument('--flag', action='store_true', help="A flag option (no argument)")

# 解析命令行输入
args = parser.parse_args()

# 使用参数
print(f"Number: {args.number}")
print(f"Float: {args.float}")
print(f"Flag: {args.flag}")

