import requests
import os

# Target file
file_path = "models/pts_in_hull.npy"

# NEW RELIABLE MIRROR (Intel OpenVINO Storage)
url = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"

print(f"Attempting to fix: {file_path}")

# 1. Delete the corrupt file if it exists
if os.path.exists(file_path):
    os.remove(file_path)
    print("🗑️ Deleted corrupt file.")

# 2. Download fresh copy
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    print("⬇️ Downloading fresh copy from Intel Mirror...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print("✅ Success! New pts_in_hull.npy created.")
        print(f"📁 File size: {os.path.getsize(file_path)} bytes (Should be ~5000 bytes)")
    else:
        print(f"❌ Failed. Server returned code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")