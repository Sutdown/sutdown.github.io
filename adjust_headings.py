#!/usr/bin/env python3
"""
批量修改文章标题层级
根据每个文件的第一个标题层级，统一调整所有标题
使第一个标题变为##
"""
import os
import re

# 定义要处理的目录
CONTENT_DIR = r'e:\project\SutdownBlog\content\post\ai'

# 目标标题层级
TARGET_LEVEL = 2  # ##

# 匹配标题的正则表达式
HEADING_REGEX = r'^(#{1,6})(\s+)(.*)$'

def fix_headings(file_path):
    """修复单个文件的标题层级"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到第一个标题
    first_heading = re.search(HEADING_REGEX, content, flags=re.MULTILINE)
    if not first_heading:
        print(f"未找到标题: {file_path}")
        return
    
    # 获取第一个标题的层级
    current_level = len(first_heading.group(1))
    
    # 计算需要调整的层级差
    level_diff = TARGET_LEVEL - current_level
    
    # 如果层级已经符合要求，无需修改
    if level_diff == 0:
        print(f"层级正确: {file_path}")
        return
    
    def adjust_heading(match):
        """调整单个标题的层级"""
        hashes = match.group(1)
        space = match.group(2)
        text = match.group(3)
        
        current_hashes = len(hashes)
        new_hashes = max(1, current_hashes + level_diff)  # 确保至少有一个#
        
        return f"{'#' * new_hashes}{space}{text}"
    
    # 调整所有标题
    fixed_content = re.sub(HEADING_REGEX, adjust_heading, content, flags=re.MULTILINE)
    
    # 保存文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"已修复: {file_path} (将层级从 {current_level} 调整为 {TARGET_LEVEL})")

def main():
    """主函数"""
    print("开始修复文章标题层级...")
    print(f"目标标题层级: {TARGET_LEVEL} (#{'#' * TARGET_LEVEL})")
    print("=" * 50)
    
    # 遍历目录中的所有markdown文件
    for filename in os.listdir(CONTENT_DIR):
        if filename.endswith('.md'):
            file_path = os.path.join(CONTENT_DIR, filename)
            fix_headings(file_path)
    
    print("=" * 50)
    print("修复完成!")

if __name__ == '__main__':
    main()