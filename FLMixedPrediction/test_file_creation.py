from pathlib import Path
import os

# 打印当前工作目录
print(f'Current working directory: {os.getcwd()}')

# 尝试使用绝对路径创建文件
project_root = Path('/home/plasmid/Project/FLMixedPrediction')
print(f'Project root: {project_root}')

# 创建测试目录
test_dir = project_root / 'test_dir'
test_dir.mkdir(exist_ok=True)
print(f'Created test directory: {test_dir}')
print(f'Test directory exists: {test_dir.exists()}')

# 创建测试文件
test_file = test_dir / 'test.txt'
test_file.write_text('Test content')
print(f'Created test file: {test_file}')
print(f'Test file exists: {test_file.exists()}')

# 读取测试文件内容
if test_file.exists():
    content = test_file.read_text()
    print(f'Test file content: {content}')

# 列出目录内容
print(f'\nTest directory contents: {list(test_dir.glob("*"))}')

print('Test completed!')
