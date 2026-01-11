#!/usr/bin/env python3
import os
import hashlib
import urllib.request
from pathlib import Path

CONTENT_DIR = 'content/post'
STATIC_IMAGES_DIR = 'static/images'
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400
API_URL = 'https://api.sretna.cn/api/anime.php'

def generate_image_slug(title, date):
    combined = f"{title}{date}"
    return hashlib.md5(combined.encode()).hexdigest()[:8] + '.jpg'

def download_image(url, filepath):
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded: {filepath}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def process_post(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split front matter
    if not content.startswith('---'):
        return

    parts = content.split('---', 2)
    if len(parts) < 3:
        return

    front_matter_str = parts[1]
    body = parts[2]

    # Check if image already exists
    if 'image:' in front_matter_str:
        print(f"Image already exists in {md_file}")
        return

    # Extract title and date for slug
    title = 'Untitled'
    date = '2024-01-01'
    for line in front_matter_str.strip().split('\n'):
        if line.startswith('title:'):
            title = line.split(':', 1)[1].strip().strip('"').strip("'")
        elif line.startswith('date:'):
            date = line.split(':', 1)[1].strip().strip('"').strip("'")

    image_slug = generate_image_slug(title, str(date))
    image_path = Path(STATIC_IMAGES_DIR) / image_slug

    if not image_path.exists():
        url = API_URL
        download_image(url, image_path)

    # Add image line to front matter
    new_front_matter = front_matter_str.rstrip() + f"\nimage: /images/{image_slug}\n"
    new_content = f"---{new_front_matter}---{body}"

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Updated: {md_file}")

def main():
    Path(STATIC_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    for md_file in Path(CONTENT_DIR).rglob('*.md'):
        if md_file.name == '_index.md':
            continue
        process_post(md_file)

if __name__ == '__main__':
    main()