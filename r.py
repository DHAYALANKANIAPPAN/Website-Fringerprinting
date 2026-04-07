"""
Automated Website Traffic Capture
Captures PCAPs for multiple websites with proper labeling
"""

import subprocess
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import threading

# Configuration
CAPTURE_DURATION = 30  # seconds per website
WEBSITES = {
    'bbc_news': 'https://www.bbc.com/news',
    'coursera': 'https://www.coursera.org',
    'discord': 'https://discord.com',
    'github': 'https://github.com',
    'stackoverflow': 'https://stackoverflow.com',
    'reddit': 'https://www.reddit.com',
    'wikipedia': 'https://www.wikipedia.org',
    'youtube': 'https://www.youtube.com',
    'amazon': 'https://www.amazon.com',
    'netflix': 'https://www.netflix.com'
}

def capture_pcap(website_name, url, duration=30):
    """Capture network traffic for a specific website"""
    
    # Create directory if it doesn't exist
    os.makedirs(f'pcaps/{website_name}', exist_ok=True)
    
    # PCAP filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pcap_file = f'pcaps/{website_name}/{website_name}_{timestamp}.pcap'
    
    print(f"\n📡 Capturing {website_name}...")
    print(f"   URL: {url}")
    print(f"   Duration: {duration} seconds")
    
    # Start tshark capture
    tshark_cmd = [
        'tshark', '-i', 'Wi-Fi',  # Change 'Wi-Fi' to your network interface
        '-a', f'duration:{duration}',
        '-w', pcap_file,
        '-f', 'not port 22'  # Exclude SSH traffic
    ]
    
    # Start capture process
    capture_process = subprocess.Popen(
        tshark_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Start browser
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load website
        driver.get(url)
        time.sleep(5)  # Wait for initial load
        
        # Scroll and interact to generate more traffic
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(2)
        
        # Wait for remaining capture time
        remaining = max(0, duration - 15)
        time.sleep(remaining)
        
    except Exception as e:
        print(f"   ⚠️ Browser error: {e}")
    
    finally:
        driver.quit()
        capture_process.wait()
    
    # Check if PCAP has enough packets
    try:
        result = subprocess.run(
            ['tshark', '-r', pcap_file, '-z', 'io,stat,1'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Count packets
        packet_count = subprocess.run(
            ['tshark', '-r', pcap_file, '-q', '-z', 'io,stat,0'],
            capture_output=True,
            text=True
        )
        
        print(f"   ✅ Captured: {pcap_file}")
        
    except:
        print(f"   ⚠️ PCAP may be empty")
    
    return pcap_file

def capture_multiple_websites():
    """Capture traffic for multiple websites"""
    
    print("="*60)
    print("Website Traffic Capture Tool")
    print("="*60)
    print("\n⚠️  Requirements:")
    print("   - Wireshark/tshark installed")
    print("   - Chrome browser with chromedriver")
    print("   - Run as administrator (Windows) or sudo (Linux/Mac)")
    print("\n" + "="*60)
    
    # Check if tshark is available
    try:
        subprocess.run(['tshark', '--version'], capture_output=True, check=True)
    except:
        print("\n❌ tshark not found! Please install Wireshark/tshark")
        return
    
    # Select websites to capture
    print("\nAvailable websites:")
    for i, (name, url) in enumerate(WEBSITES.items(), 1):
        print(f"  {i}. {name} - {url}")
    
    print("\nOptions:")
    print("  a - Capture ALL websites")
    print("  q - Quit")
    
    choice = input("\nSelect websites (comma-separated numbers or 'a'): ").strip()
    
    if choice.lower() == 'q':
        return
    
    websites_to_capture = []
    
    if choice.lower() == 'a':
        websites_to_capture = list(WEBSITES.items())
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            websites_to_capture = [list(WEBSITES.items())[i-1] for i in indices if 1 <= i <= len(WEBSITES)]
        except:
            print("Invalid selection")
            return
    
    # Capture each website
    for website_name, url in websites_to_capture:
        capture_pcap(website_name, url, CAPTURE_DURATION)
        time.sleep(2)  # Pause between captures
    
    print("\n" + "="*60)
    print("✅ Capture complete!")
    print(f"PCAPs saved in 'pcaps/' folder")
    print("="*60)

if __name__ == "__main__":
    capture_multiple_websites()
    