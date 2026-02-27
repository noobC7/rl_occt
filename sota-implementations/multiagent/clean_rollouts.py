import os

def delete_large_pt_files(root_dir, size_threshold_mb=50):
    """
    递归删除指定目录下超过指定大小的.pt文件
    
    Args:
        root_dir: 要扫描的根目录
        size_threshold_mb: 文件大小阈值（MB），默认100MB
    """
    # 将MB转换为字节（1MB = 1024*1024 字节）
    size_threshold = size_threshold_mb * 1024 * 1024
    
    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：目录 {root_dir} 不存在！")
        return
    
    # 递归遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 筛选.pt文件
            if filename.lower().endswith('.pt'):
                file_path = os.path.join(dirpath, filename)
                try:
                    # 获取文件大小（字节）
                    file_size = os.path.getsize(file_path)
                    
                    # 判断是否超过阈值
                    if file_size > size_threshold:
                        # 删除文件
                        os.remove(file_path)
                        print(f"已删除超大文件：{file_path} | 大小：{file_size/1024/1024:.2f} MB")
                except PermissionError:
                    print(f"权限不足，无法删除：{file_path}")
                except FileNotFoundError:
                    print(f"文件已被删除，跳过：{file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{str(e)}")

if __name__ == "__main__":
    # 替换为你要扫描的目录路径（绝对路径/相对路径均可）
    # Windows示例：r"C:\Users\yourname\projects"
    # Linux/macOS示例："/home/yourname/projects"
    target_directory = "/home/yons/Graduation/rl_occt/outputs"
    delete_large_pt_files(target_directory)