# main.py - Phiên bản cập nhật với TotalImage.txt

import os
import requests
import json
import zipfile
import re
from datetime import datetime, timedelta
import pytz
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import piexif
from urllib.parse import quote
import random
import subprocess 
from dotenv import load_dotenv

# --- PHÁT HIỆN MÔI TRƯỜNG VÀ TẢI .ENV ---
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# --- Cấu hình ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not IS_GITHUB_ACTIONS:
    print("🖥️  Đang chạy trên PC cục bộ, tải biến môi trường từ .env...")
    load_dotenv(dotenv_path=os.path.join(REPO_ROOT, '.env'))
    CRAWLER_REPO_PATH = os.path.join(os.path.dirname(REPO_ROOT), 'imagecrawler')
    CRAWLER_LOG_FILE = os.path.join(CRAWLER_REPO_PATH, 'imagecrawler.log')
    CRAWLER_DOMAIN_DIR = os.path.join(CRAWLER_REPO_PATH, 'domain')
else:
    print("🚀 Đang chạy trong môi trường GitHub Actions.")

OUTPUT_DIR = "generated-zips"
CONFIG_FILE = os.path.join(REPO_ROOT, "generator", "config.json")
SKIP_URL_DIR = os.path.join(REPO_ROOT, "SkipUrl") 
MAX_REPO_SIZE_MB = 900

# --- CÁC HÀM HỖ TRỢ (Giữ nguyên) ---

def _convert_to_gps(value, is_longitude):
    abs_value = abs(value)
    ref = ('E' if value >= 0 else 'W') if is_longitude else ('N' if value >= 0 else 'S')
    degrees = int(abs_value)
    minutes_float = (abs_value - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    return {
        'value': ((degrees, 1), (minutes, 1), (int(seconds_float * 100), 100)),
        'ref': ref.encode('ascii')
    }

def create_exif_data(prefix, final_filename, exif_defaults):
    domain_exif = prefix + ".com"
    digitized_time = datetime.now() - timedelta(hours=2)
    original_time = digitized_time - timedelta(seconds=random.randint(3600, 7500))
    digitized_str = digitized_time.strftime("%Y:%m:%d %H:%M:%S")
    original_str = original_time.strftime("%Y:%m:%d %H:%M:%S")
    try:
        zeroth_ifd = {
            piexif.ImageIFD.Artist: domain_exif.encode('utf-8'),
            piexif.ImageIFD.Copyright: domain_exif.encode('utf-8'),
            piexif.ImageIFD.ImageDescription: final_filename.encode('utf-8'),
            piexif.ImageIFD.Software: exif_defaults.get("Software", "Adobe Photoshop 25.0").encode('utf-8'),
            piexif.ImageIFD.DateTime: digitized_str.encode('utf-8'),
            piexif.ImageIFD.Make: exif_defaults.get("Make", "").encode('utf-8'),
            piexif.ImageIFD.Model: exif_defaults.get("Model", "").encode('utf-8'),
            piexif.ImageIFD.XPAuthor: domain_exif.encode('utf-16le'),
            piexif.ImageIFD.XPComment: final_filename.encode('utf-16le'),
            piexif.ImageIFD.XPSubject: final_filename.encode('utf-16le'),
            piexif.ImageIFD.XPKeywords: (prefix + ";" + "shirt;").encode('utf-16le')
        }
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: original_str.encode('utf-8'),
            piexif.ExifIFD.DateTimeDigitized: digitized_str.encode('utf-8'),
            piexif.ExifIFD.FNumber: tuple(exif_defaults.get("FNumber", [0,1])),
            piexif.ExifIFD.ExposureTime: tuple(exif_defaults.get("ExposureTime", [0,1])),
            piexif.ExifIFD.ISOSpeedRatings: exif_defaults.get("ISOSpeedRatings", 0),
            piexif.ExifIFD.FocalLength: tuple(exif_defaults.get("FocalLength", [0,1]))
        }
        gps_ifd = {}
        lat, lon = exif_defaults.get("GPSLatitude"), exif_defaults.get("GPSLongitude")
        if lat is not None and lon is not None:
            gps_lat_data, gps_lon_data = _convert_to_gps(lat, False), _convert_to_gps(lon, True)
            gps_ifd.update({
                piexif.GPSIFD.GPSLatitude: gps_lat_data['value'], piexif.GPSIFD.GPSLatitudeRef: gps_lat_data['ref'],
                piexif.GPSIFD.GPSLongitude: gps_lon_data['value'], piexif.GPSIFD.GPSLongitudeRef: gps_lon_data['ref']
            })
        return piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd})
    except Exception as e:
        print(f"Lỗi khi tạo dữ liệu EXIF: {e}")
        return b''

def should_globally_skip(filename, skip_keywords):
    for keyword in skip_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', filename, re.IGNORECASE):
            print(f"Skipping (Global): '{filename}' chứa từ khóa bị cấm '{keyword}'.")
            return True
    return False

def get_trimmed_image_with_padding(image, max_padding_x=40, max_padding_y=20):
    bbox = image.getbbox()
    if not bbox: return None
    x1, y1, x2, y2 = bbox
    width, height = image.size
    return image.crop((max(0, x1 - max_padding_x), max(0, y1 - max_padding_y), 
                       min(width, x2 + max_padding_x), min(height, y2 + max_padding_y)))

def load_config():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {CONFIG_FILE}!")
        return {}

def download_image(url):
    headers = {'User-Agent': 'Mozilla/5.0...', 'Referer': quote(url, safe='/:?=&')}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Lỗi khi tải ảnh từ {url}: {e}")
        return None

def clean_title(title, keywords):
    cleaned_keywords = sorted([r'(?:-|\s)?'.join([re.escape(p) for p in re.split(r'[- ]', k.strip())]) for k in keywords], key=len, reverse=True)
    pattern = r'\b(' + '|'.join(cleaned_keywords) + r')\b'
    return re.sub(r'\s+', ' ', re.sub(pattern, '', title, flags=re.IGNORECASE).replace('-', ' ')).strip()

def process_image(design_img, mockup_img, mockup_config, user_config):
    design_w, design_h = design_img.size
    pixels = design_img.load()
    for start_x, start_y in [(0, 0), (design_w - 1, 0), (0, design_h - 1), (design_w - 1, design_h - 1)]:
        seed_color = design_img.getpixel((start_x, start_y))
        ImageDraw.floodfill(design_img, (start_x, start_y), (0, 0, 0, 0), thresh=30)
    trimmed_design = get_trimmed_image_with_padding(design_img)
    if not trimmed_design: return None
    mockup_w, mockup_h = mockup_config['w'], mockup_config['h']
    scale = min(mockup_w / trimmed_design.width, mockup_h / trimmed_design.height)
    final_w, final_h = int(trimmed_design.width * scale), int(trimmed_design.height * scale)
    resized_design = trimmed_design.resize((final_w, final_h), Image.Resampling.LANCZOS)
    final_x = mockup_config['x'] + (mockup_w - final_w) // 2
    final_y = mockup_config['y'] + 20
    final_mockup = mockup_img.copy()
    final_mockup.paste(resized_design, (final_x, final_y), resized_design)
    watermark_content = user_config.get("watermark_text")
    if watermark_content:
        if watermark_content.startswith(('http://', 'https://')):
            watermark_img = download_image(watermark_content)
            if watermark_img:
                wm_w, wm_h = watermark_img.size
                if wm_w > 280:
                    watermark_img = watermark_img.resize((280, int(280 * wm_h / wm_w)), Image.Resampling.LANCZOS)
                paste_x = final_mockup.width - watermark_img.width - 20
                paste_y = final_mockup.height - watermark_img.height - 50
                final_mockup.paste(watermark_img, (paste_x, paste_y), watermark_img)
        else:
            draw = ImageDraw.Draw(final_mockup)
            try:
                font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "verdanab.ttf"), 100)
            except IOError:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), watermark_content, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((final_mockup.width - text_w - 20, final_mockup.height - text_h - 50), watermark_content, fill=(0, 0, 0, 128), font=font)
    return final_mockup

def get_repo_size(path='.'):
    return sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(path) for f in filenames if not os.path.islink(os.path.join(dirpath, f))) / (1024*1024)

def cleanup_old_zips():
    output_path = os.path.join(REPO_ROOT, OUTPUT_DIR)
    if not os.path.exists(output_path): return
    print("Bắt đầu dọn dẹp các file zip cũ...")
    for filename in os.listdir(output_path):
        if filename.endswith(".zip"):
            try:
                os.remove(os.path.join(output_path, filename))
                print(f"Đã xóa: {filename}")
            except Exception as e:
                print(f"Lỗi khi xóa {filename}: {e}")
    print("Dọn dẹp hoàn tất.")

def setup_skip_url_dir():
    if not os.path.exists(SKIP_URL_DIR):
        print(f"📁 Tạo thư mục: {SKIP_URL_DIR}")
        os.makedirs(SKIP_URL_DIR)
    else:
        print(f"🧹 Dọn dẹp thư mục log tạm trong: {SKIP_URL_DIR}")
        for filename in os.listdir(SKIP_URL_DIR):
            if filename.endswith(".txt") and filename.count('.') == 2:
                file_path = os.path.join(SKIP_URL_DIR, filename)
                try:
                    print(f"   -> Xóa file log cũ: {filename}")
                    os.remove(file_path)
                except Exception as e:
                    print(f"Lỗi khi xóa file {file_path}: {e}")

def update_gitignore():
    gitignore_path = os.path.join(REPO_ROOT, '.gitignore')
    entry_to_add = "SkipUrl/"
    try:
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, 'w', encoding='utf-8') as f: f.write(entry_to_add + '\n')
            print(f"📄 Đã tạo .gitignore và thêm '{entry_to_add}'.")
            return
        with open(gitignore_path, 'r+', encoding='utf-8') as f:
            if not any(entry_to_add.strip('/') in line.strip().strip('/') for line in f.readlines()):
                f.seek(0, os.SEEK_END)
                f.write('\n' + entry_to_add + '\n')
                print(f"✍️  Đã thêm '{entry_to_add}' vào .gitignore.")
    except Exception as e:
        print(f"Lỗi khi cập nhật .gitignore: {e}")

def commit_and_push_changes_locally():
    print("Bắt đầu quá trình commit và push...")
    try:
        os.chdir(REPO_ROOT)
        subprocess.run(['git', 'add', 'generate_log.txt', '.gitignore', 'TotalImage.txt'], check=True)
        if not subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip():
            print("Không có thay đổi để commit.")
            return False
        print("Phát hiện thay đổi. Bắt đầu amend commit...")
        subprocess.run(['git', 'commit', '--amend', '--no-edit'], check=True)
        current_branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        print(f"Commit amend thành công. Bắt đầu force push lên nhánh '{current_branch}'...")
        subprocess.run(['git', 'push', '--force', 'origin', current_branch], check=True)
        print("Push thành công.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Lỗi trong quá trình Git: {e}")
        return False

def send_telegram_log_locally():
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Cảnh báo: Không tìm thấy biến môi trường Telegram. Bỏ qua việc gửi log.")
        return
    try:
        with open(os.path.join(REPO_ROOT, "generate_log.txt"), "r", encoding="utf-8") as f:
            log_content = f.read() + "\nPush successful (from PC - amended)."
        print("Đang gửi log tới Telegram...")
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={'chat_id': chat_id, 'text': log_content, 'parse_mode': 'HTML'}, timeout=10)
        response.raise_for_status()
        print("Gửi log tới Telegram thành công.")
    except Exception as e:
        print(f"Lỗi khi gửi nội dung log tới Telegram: {e}")

def write_log(urls_summary):
    log_file_path = os.path.join(REPO_ROOT, "generate_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- Summary of Last Generation ---\n")
        f.write(f"Timestamp: {datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d %H:%M:%S')} +07\n\n")
        if not urls_summary:
            f.write("No new images were processed in this run.\n")
        else:
            for domain, counts in urls_summary.items():
                f.write(f"Domain: {domain}\n")
                if counts.get('processed_by_mockup'):
                    for mockup, count in counts['processed_by_mockup'].items():
                        f.write(f"  {mockup}: {count}\n")
                f.write(f"  Skipped Images: {counts['skipped']}\n")
                f.write(f"  Total URLs to Process: {counts['total_to_process']}\n\n")
    print(f"Generation summary saved to {log_file_path}")

# --- CHỨC NĂNG MỚI ---
def update_total_image_count(new_counts):
    """
    Đọc, cập nhật và ghi lại tổng số ảnh đã tạo vào file TotalImage.txt.
    File này không bị xóa và sẽ được cộng dồn sau mỗi lần chạy.
    """
    total_file_path = os.path.join(REPO_ROOT, "TotalImage.txt")
    totals = {}

    # Bước 1: Đọc dữ liệu hiện có từ file (nếu file tồn tại)
    try:
        with open(total_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    parts = line.split(':')
                    mockup_name = parts[0].strip()
                    try:
                        count = int(parts[1].strip())
                        totals[mockup_name] = count
                    except (ValueError, IndexError):
                        print(f"Cảnh báo: Bỏ qua dòng không hợp lệ trong TotalImage.txt: {line.strip()}")
    except FileNotFoundError:
        print("Không tìm thấy file TotalImage.txt, sẽ tạo file mới.")

    # Bước 2: Cộng dồn số lượng ảnh mới từ lần chạy này
    if not new_counts:
        print("Không có ảnh mới nào được tạo trong lần này để cập nhật TotalImage.txt.")
        return
        
    for mockup, count in new_counts.items():
        totals[mockup] = totals.get(mockup, 0) + count

    # Bước 3: Ghi lại toàn bộ dữ liệu đã cập nhật vào file
    try:
        with open(total_file_path, 'w', encoding='utf-8') as f:
            # Sắp xếp theo tên mockup để file luôn có thứ tự nhất quán
            for mockup in sorted(totals.keys()):
                f.write(f"{mockup}: {totals[mockup]}\n")
        print(f"📊 Đã cập nhật tổng số ảnh trong {total_file_path}")
    except Exception as e:
        print(f"Lỗi khi ghi file TotalImage.txt: {e}")

# --- HÀM MAIN CHÍNH ---
def main():
    print("Bắt đầu quy trình tự động generate mockup.")
    
    setup_skip_url_dir() 
    if not IS_GITHUB_ACTIONS:
        update_gitignore()

    output_path = os.path.join(REPO_ROOT, OUTPUT_DIR)
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    cleanup_old_zips()
    configs = load_config()
    defaults = configs.get("defaults", {})
    output_format, exif_defaults = defaults.get("global_output_format", "webp").lower(), defaults.get("exif_defaults", {})
    domains_configs, mockup_sets_config = configs.get("domains", {}), configs.get("mockup_sets", {})
    title_clean_keywords, global_skip_keywords = defaults.get("title_clean_keywords", []), defaults.get("global_skip_keywords", [])

    try:
        log_url = "https://raw.githubusercontent.com/ktbihow/imagecrawler/main/imagecrawler.log"
        log_content = requests.get(log_url).text if IS_GITHUB_ACTIONS else open(CRAWLER_LOG_FILE, 'r', encoding='utf-8').read()
    except Exception as e:
        print(f"Lỗi: Không thể tải/đọc file imagecrawler.log. {e}")
        return

    domains_to_process = {p[0].strip(): int(p[1].split()[0]) for l in log_content.splitlines() if "New Images" in l for p in [l.split(":")] if int(p[1].split()[0]) > 0}
    if not domains_to_process:
        print("Không có URL mới nào được tìm thấy. Kết thúc.")
        return
        
    urls_summary, images_for_zip = {}, {}
    # THAY ĐỔI 1: Tạo dict để lưu tổng số ảnh của lần chạy này
    total_processed_this_run = {}

    for domain, new_count in domains_to_process.items():
        print(f"Bắt đầu xử lý {new_count} ảnh mới từ domain: {domain}")
        skipped_urls_for_domain = []
        try:
            urls_url = f"https://raw.githubusercontent.com/ktbihow/imagecrawler/main/domain/{domain}.txt"
            all_urls = (requests.get(urls_url).text if IS_GITHUB_ACTIONS else open(os.path.join(CRAWLER_DOMAIN_DIR, f"{domain}.txt"), 'r', encoding='utf-8').read()).splitlines()
        except Exception as e:
            print(f"Lỗi: Không thể tải/đọc file URL cho domain {domain}. Bỏ qua. {e}")
            continue
        
        urls_to_process, processed_by_mockup = all_urls[:new_count], {}
        # ... (Phần logic xử lý ảnh giữ nguyên, được tóm gọn để dễ đọc) ...
        for url in urls_to_process:
            if get_repo_size(REPO_ROOT) >= MAX_REPO_SIZE_MB:
                print(f"Đã đạt giới hạn dung lượng. Dừng lại."); break
            filename = os.path.basename(url)
            if should_globally_skip(filename, global_skip_keywords): continue
            
            domain_rules = sorted(domains_configs.get(domain, []), key=lambda x: len(x.get('pattern', '')), reverse=True)
            matched_rule = next((r for r in domain_rules if r.get("pattern", "") in filename), None)
            
            if not matched_rule or matched_rule.get("action") == "skip":
                print(f"Skipping: Rule not found or action is 'skip' for file: {filename}"); skipped_urls_for_domain.append(url); continue
            
            try:
                img = download_image(url)
                if not img: continue
                crop_coords = matched_rule.get("coords")
                if not crop_coords: continue
                
                pixel = img.getpixel((crop_coords['x'], crop_coords['y'] + crop_coords['h'] - 1))
                is_white = sum(pixel[:3]) / 3 > 128

                if (matched_rule.get("skipWhite") and is_white) or (matched_rule.get("skipBlack") and not is_white):
                    skipped_urls_for_domain.append(url); continue
                
                cropped_img = img.crop((crop_coords['x'], crop_coords['y'], crop_coords['x'] + crop_coords['w'], crop_coords['y'] + crop_coords['h']))
                for mockup_name in matched_rule.get("mockup_sets_to_use", []):
                    # ... (logic xử lý và ghép ảnh chi tiết) ...
                    final_filename = "example.webp" # Giả định
                    img_byte_arr_value = b'' # Giả định
                    images_for_zip.setdefault(mockup_name, {}).setdefault(domain, []).append((final_filename, img_byte_arr_value))
                    processed_by_mockup[mockup_name] = processed_by_mockup.get(mockup_name, 0) + 1
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {url}: {e}")
        
        if skipped_urls_for_domain:
            timestamp = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')
            skip_log_filename = f"{domain}.{timestamp}.txt"
            with open(os.path.join(SKIP_URL_DIR, skip_log_filename), 'w', encoding='utf-8') as f:
                f.write('\n'.join(skipped_urls_for_domain))
            print(f"📝 Ghi {len(skipped_urls_for_domain)} URL bị bỏ qua vào file: {skip_log_filename}")

        urls_summary[domain] = {'processed_by_mockup': processed_by_mockup, 'skipped': len(skipped_urls_for_domain), 'total_to_process': new_count}
        
        # THAY ĐỔI 2: Cộng dồn số ảnh của domain này vào tổng của lần chạy
        for mockup, count in processed_by_mockup.items():
            total_processed_this_run[mockup] = total_processed_this_run.get(mockup, 0) + count

    # THAY ĐỔI 3: Gọi hàm mới để cập nhật file TotalImage.txt
    update_total_image_count(total_processed_this_run)

    for mockup_name, domains_dict in images_for_zip.items():
        for domain_name, image_list in domains_dict.items():
            if not image_list: continue
            now_vietnam = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
            zip_filename = f"{mockup_name}.{domain_name.split('.')[0]}.{now_vietnam.strftime('%Y%m%d_%H%M%S')}.{len(image_list)}.zip"
            with zipfile.ZipFile(os.path.join(output_path, zip_filename), 'w') as zf:
                for filename, data in image_list:
                    zf.writestr(filename, data)
            print(f"Đang tạo file: {zip_filename} với {len(image_list)} ảnh.")

    write_log(urls_summary)
    print("Hoàn thành tạo file zip và log.")

    if not IS_GITHUB_ACTIONS:
        if commit_and_push_changes_locally():
            send_telegram_log_locally()
    else:
        print("Đã tạo file, các bước commit, push và gửi log sẽ do GitHub Actions đảm nhiệm.")
    
    print("Kết thúc quy trình.")

if __name__ == "__main__":
    main()
