import os
import json
import requests
from urllib.parse import urlencode
from typing import Optional, List, Dict, Any


def download_baidu_images(keyword: str, save_dir: str = 'baidu_images', max_page: int = 3) -> int:
    """Download images from Baidu Image API.
    
    Args:
        keyword: Search keyword (e.g., 'QR code')
        save_dir: Directory to save downloaded images
        max_page: Maximum number of pages to crawl (each page contains ~30 images)
        
    Returns:
        Total number of downloaded images
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Baidu Image API base URL
    base_url = 'https://image.baidu.com/search/acjson'

    # Request headers (simulate browser to avoid being blocked)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
        'Referer': 'https://image.baidu.com/',  # Important: need to include referer
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }

    total_images = 0  # Count total downloads

    for page in range(max_page):
        # API parameters (pn is page number, starting from 0, 30 images per page)
        params = {
            'tn': 'resultjson_com',
            'logid': '7618233718212266687',  # Fixed value (can be copied from browser request)
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'fr': '',
            'word': keyword,  # Search keyword
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': '',
            'hd': '',
            'latest': '',
            'copyright': '',
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'expermode': '',
            'nojc': '',
            'isAsync': '',
            'pn': page * 30,  # Page number (0, 30, 60...)
            'rn': 30,  # 30 images per page
            'gsm': f'{page * 30:02x}',  # Hexadecimal page number (e.g., page 0 is 00, page 30 is 1e)
        }

        try:
            # Send API request
            response = requests.get(
                url=base_url,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()  # Check if request was successful

            # Parse JSON (note: Baidu returns JSON with extra characters that need processing)
            json_data = response.text.lstrip('/*baidu*/')  # Remove leading comment
            data = json.loads(json_data)

            # Extract image links (image URLs are in the 'data' field)
            images = data.get('data', [])
            if not images:
                print(f"No images found on page {page + 1}, may have reached maximum pages")
                break

            # Download images on current page
            for i, img_info in enumerate(images):
                # Image links may be in thumbURL or middleURL
                img_url = img_info.get('thumbURL') or img_info.get('middleURL')
                if not img_url:
                    continue  # Skip images without links

                total_images += 1
                try:
                    # Download image
                    img_response = requests.get(
                        img_url,
                        headers=headers,
                        timeout=10,
                        stream=True
                    )
                    img_response.raise_for_status()

                    # Generate filename (avoid duplicates)
                    ext = os.path.splitext(img_url)[1].lower()
                    if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                        ext = '.jpg'
                    filename = f'baidu_{total_images}{ext}'
                    save_path = os.path.join(save_dir, filename)

                    # Save image
                    with open(save_path, 'wb') as f:
                        for chunk in img_response.iter_content(1024):
                            if chunk:
                                f.write(chunk)
                    print(f"Downloaded {total_images}: {save_path}")

                except Exception as e:
                    print(f"Failed to download image {total_images}: {str(e)}")

        except Exception as e:
            print(f"Failed to request page {page + 1}: {str(e)}")
            continue

    print(f"All images downloaded. Total: {total_images} images saved to {save_dir}")
    return total_images


if __name__ == "__main__":
    """Main entry point for the script."""
    # Download images with keyword "QR code", 3 pages (about 90 images)
    download_baidu_images(
        keyword='二维码',  # Search for QR code images
        save_dir='baidu_images',
        max_page=3  # Can modify the number of pages (more pages = more images)
    )