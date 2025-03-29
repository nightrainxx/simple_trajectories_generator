import os
import datetime

def get_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def generate_directory_tree(startpath, ignore_dirs=['.git', '__pycache__', '.pytest_cache']):
    """生成目录树"""
    tree = []
    for root, dirs, files in os.walk(startpath):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level)
        tree.append(f'{indent}├── {os.path.basename(root)}/')
        
        subindent = '│   ' * (level + 1)
        for f in files:
            if f.endswith('.py') or f.endswith('.md') or f == 'requirements.txt':
                tree.append(f'{subindent}├── {f}')
    
    return '\n'.join(tree)

def collect_project_info():
    """收集项目信息并保存到文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, 'project_documentation.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入时间戳
        f.write(f"项目文档生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入项目结构
        f.write("="*80 + "\n")
        f.write("项目结构树:\n")
        f.write("="*80 + "\n")
        f.write(generate_directory_tree(current_dir))
        f.write("\n\n")
        
        # 收集并写入所有代码文件内容
        for root, _, files in os.walk(current_dir):
            for file in files:
                if file.endswith('.py') or file.endswith('.md') or file == 'requirements.txt':
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, current_dir)
                    
                    f.write("="*80 + "\n")
                    f.write(f"文件: {relative_path}\n")
                    f.write("="*80 + "\n")
                    f.write(get_file_content(file_path))
                    f.write("\n\n")

if __name__ == "__main__":
    collect_project_info()
    print("项目文档已生成完成，请查看 project_documentation.txt 文件") 