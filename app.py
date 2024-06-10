import os
from playwright.async_api import async_playwright
from PIL import Image
from io import BytesIO
import aiohttp
from bs4 import BeautifulSoup
import urllib.parse
import cv2
import numpy as np
import gradio as gr
import asyncio
import re
import pandas as pd

# 이미지 저장 경로 설정
images_folder = 'naver_map_images'
os.makedirs(images_folder, exist_ok=True)

def convert_naver_map_url(url):
    match = re.search(r'place/(\d+)', url)
    if match:
        place_id = match.group(1)
        return f'https://pcmap.place.naver.com/place/{place_id}/feed?from=map&fromPanelNum=1'
    else:
        raise ValueError("Invalid Naver Map URL")

async def fetch_image(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        img_data = await response.read()
        img = Image.open(BytesIO(img_data))

        # 이미지를 RGB 모드로 변환
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        return img

async def download_images(url):
    converted_url = convert_naver_map_url(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(converted_url)

        # 특정 요소가 로드될 때까지 대기 (여기서는 img 태그를 예로 듦)
        await page.wait_for_selector('img')

        content = await page.content()

        await browser.close()

    async with aiohttp.ClientSession() as session:
        soup = BeautifulSoup(content, 'html.parser')
        images = soup.find_all('img')

        tasks = []
        for i, img in enumerate(images):
            img_url = img.get('src')
            if img_url and img_url.startswith('http'):
                # URL 디코딩 처리
                img_url = urllib.parse.unquote(img_url)
                tasks.append(fetch_image(session, img_url))

        downloaded_images = await asyncio.gather(*tasks)
        return downloaded_images

def calculate_image_similarity(img1, img2):
    img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    img1_gray = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))

    difference = cv2.absdiff(img1_gray, img2_gray)

    similarity = 1 - (np.sum(difference) / (img1_gray.shape[0] * img1_gray.shape[1] * 255))
    return similarity

def find_similar_images(reference_image, downloaded_images, threshold=0.98):
    similar_images = []
    for i, img in enumerate(downloaded_images):
        similarity = calculate_image_similarity(reference_image, img)
        if similarity >= threshold:
            similar_images.append((f"Image {i}", similarity))
            print(f"Found similar image: Image {i} with similarity: {similarity}")

    return similar_images

async def run_download_and_compare(image, urls_file):
    urls = []
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()

    results = []
    for url in urls:
        try:
            downloaded_images = await download_images(url)
            similar_images_found = len(find_similar_images(image, downloaded_images)) > 0
            results.append((url, "O" if similar_images_found else "X"))
        except Exception as e:
            results.append((url, str(e)))

    return results

def gradio_interface(image, urls_file):
    try:
        result = asyncio.run(run_download_and_compare(image, urls_file))
        return result
    except Exception as e:
        return [("", str(e))]

def save_results_to_csv(results):
    df = pd.DataFrame(results, columns=["URL", "유사한 이미지 여부"])
    csv_file = "results.csv"
    df.to_csv(csv_file, index=False)
    return csv_file

error_message = "An error occurred"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Image(type="pil"), gr.File(label="URLs File")],
    outputs=gr.Dataframe(headers=["URL", "유사한 이미지 여부"]),
    title="Image Similarity Finder",
    description="Upload an image and a file containing URLs (one per line) to find similar images from the downloaded sets.",
)

# 저장 버튼을 추가하고, CSV 저장 기능을 연결합니다.
def gradio_interface_with_save(image, urls_file):
    results = gradio_interface(image, urls_file)
    return results, save_results_to_csv(results)

iface_with_save = gr.Interface(
    fn=gradio_interface_with_save,
    inputs=[gr.Image(type="pil"), gr.File(label="URLs File")],
    outputs=[gr.Dataframe(headers=["URL", "유사한 이미지 여부"]), gr.File()],
    title="Image Similarity Finder with Save",
    description="Upload an image and a file containing URLs (one per line) to find similar images from the downloaded sets.",
)

iface_with_save.launch(debug=True)
