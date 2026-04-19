from pathlib import Path
import time

# 创建测试目录
print("Starting test script...")
test_dir = Path('./test_figure')
test_dir.mkdir(exist_ok=True)
print(f"Created test directory: {test_dir}")

# 创建测试文件
test_file = test_dir / "test.txt"
test_file.write_text(f"Test file created at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Created test file: {test_file}")

# 读取并打印文件内容
with open(test_file, 'r') as f:
    content = f.read()
print(f"File content: {content}")

print("Test script completed successfully!")
