import os
import re

content_dir = r"e:/project/SutdownBlog/content"

for root, dirs, files in os.walk(content_dir):
    for file in files:
        if file.endswith('.md'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content.replace('/static/img/', '/img/')
                
                if content != new_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {filepath}")
            except Exception as e:
                print(f"Error: {filepath} - {e}")

print("Done!")
