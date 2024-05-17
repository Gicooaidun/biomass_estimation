### with this file you can download the icesat granules from the links in the 
### granule_links_icesat.txt file and save them to the specified directory


import os
import getpass
import requests

def download_file(url, session, output_dir):
    """Download an individual file."""
    local_filename = url.split('/')[-1]
    output_path = os.path.join(output_dir, local_filename)
    with session.get(url, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded '{local_filename}' to '{output_path}'")
    return output_path

def download_dataset(urls_filename, output_dir):
    """Download all files listed in the given file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Login and session setup
    username = input("Enter your Earthdata Login username: ")
    password = getpass.getpass("Enter your Earthdata Login password: ")
    session = requests.Session()
    
    # Attempt to login to initiate session
    auth_url = 'https://urs.earthdata.nasa.gov/oauth/authorize'
    session.auth = (username, password)
    session.get(auth_url)  # This might not be necessary if the session automatically handles redirects and cookies

    with open(urls_filename, 'r') as file:
        for url in file:
            url = url.strip()
            if url:
                try:
                    download_file(url, session, output_dir)
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    # Path to the file containing URLs
    urls_filename = 'code/newer/links.txt'
    output_dir = 'code/newer/icesat_granules/'
    download_dataset(urls_filename, output_dir)
