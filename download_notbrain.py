import os
import requests

# Folder path
folder = "dataset/notbrain"
os.makedirs(folder, exist_ok=True)

# Random image URLs (cars, dogs, people, etc.)
urls = [
    "https://picsum.photos/300/300",
    "https://picsum.photos/301/301",
    "https://picsum.photos/302/302",
    "https://picsum.photos/303/303",
    "https://picsum.photos/304/304",
]

print("Downloading images...")

count = 0

# Download ~100 images
for i in range(100):
    try:
        url = urls[i % len(urls)]
        response = requests.get(url)

        file_path = os.path.join(folder, f"img_{i}.jpg")
        with open(file_path, "wb") as f:
            f.write(response.content)

        count += 1
        print(f"Downloaded {count} images")

    except:
        pass

print("✅ Dataset Ready in dataset/notbrain/")