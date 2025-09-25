# main.py - Phiên bản cập nhật với .env và git commit --amend

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
from dotenv import load_dotenv # THAY ĐỔI: Thêm thư viện dotenv

# --- PHÁT HIỆN MÔI TRƯỜDNG VÀ TẢI .ENV ---
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# --- Cấu hình ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not IS_GITHUB_ACTIONS:
    print("🖥️  Đang chạy trên PC cục bộ, tải biến môi trường từ .env...")
    # THAY ĐỔI: Tải các biến từ file .env στο thư mục gốc
    load_dotenv(dotenv_path=os.path.join(REPO_ROOT, '.env'))
    CRAWLER_REPO_PATH = os.path.join(os.path.dirname(REPO_ROOT), 'imagecrawler')
    CRAWLER_LOG_FILE = os.path.join(CRAWLER_REPO_PATH, 'imagecrawler.log')
    CRAWLER_DOMAIN_DIR = os.path.join(CRAWLER_REPO_PATH, 'domain')
else:
    print("🚀 Đang chạy trong môi trường GitHub Actions.")

OUTPUT_DIR = "generated-zips"
CONFIG_FILE = os.path.join(REPO_ROOT, "generator", "config.json")
MAX_REPO_SIZE_MB = 900

# --- (Toàn bộ các hàm hỗ trợ từ _convert_to_gps đến cleanup_old_zips giữ nguyên) ---
# ... (Giả sử các hàm này đã có ở đây)
def _convert_to_gps(value, is_longitude):
    """Chuyển đổi tọa độ thập phân sang định dạng EXIF GPS."""
    abs_value = abs(value)
    if is_longitude:
        ref = 'E' if value >= 0 else 'W'
    else:
        ref = 'N' if value >= 0 else 'S'
    degrees = int(abs_value)
    minutes_float = (abs_value - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    return {
        'value': ((degrees, 1), (minutes, 1), (int(seconds_float * 100), 100)),
        'ref': ref.encode('ascii')
    }

def create_exif_data(prefix, final_filename, exif_defaults):
    """Tạo chuỗi bytes EXIF."""
    domain_exif = prefix + ".com"
    digitized_time = datetime.now() - timedelta(hours=2)
    min_seconds_offset = 3600
    max_seconds_offset = 7500
    random_seconds = random.randint(min_seconds_offset, max_seconds_offset)
    original_time = digitized_time - timedelta(seconds=random_seconds)
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
        lat = exif_defaults.get("GPSLatitude")
        lon = exif_defaults.get("GPSLongitude")
        if lat is not None and lon is not None:
            gps_lat_data = _convert_to_gps(lat, is_longitude=False)
            gps_lon_data = _convert_to_gps(lon, is_longitude=True)
            gps_ifd[piexif.GPSIFD.GPSLatitude] = gps_lat_data['value']
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = gps_lat_data['ref']
            gps_ifd[piexif.GPSIFD.GPSLongitude] = gps_lon_data['value']
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = gps_lon_data['ref']
        exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd}
        return piexif.dump(exif_dict)
    except Exception as e:
        print(f"Lỗi khi tạo dữ liệu EXIF: {e}")
        return b''

def should_globally_skip(filename, skip_keywords):
    """Kiểm tra xem tên tệp có chứa từ khóa bỏ qua toàn cục hay không."""
    for keyword in skip_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, filename, re.IGNORECASE):
            print(f"Skipping (Global): '{filename}' chứa từ khóa bị cấm '{keyword}'.")
            return True
    return False

def get_trimmed_image_with_padding(image, max_padding_x=40, max_padding_y=20):
    """Cắt viền trong suốt nhưng giữ lại một khoảng đệm."""
    bbox = image.getbbox()
    if not bbox: return None
    x1, y1, x2, y2 = bbox
    width, height = image.size
    new_x1 = max(0, x1 - max_padding_x)
    new_y1 = max(0, y1 - max_padding_y)
    new_x2 = min(width, x2 + max_padding_x)
    new_y2 = min(height, y2 + max_padding_y)
    return image.crop((new_x1, new_y1, new_x2, new_y2))

def load_config():
    """Tải cấu hình từ file config.json."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {CONFIG_FILE}!")
        return {}

def download_image(url):
    """Tải ảnh từ URL."""
    safe_url_for_header = quote(url, safe='/:?=&')
    headers = {'User-Agent': 'Mozilla/5.0...', 'Referer': safe_url_for_header}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Lỗi khi tải ảnh từ {url}: {e}")
        return None

def clean_title(title, keywords):
    """Làm sạch tiêu đề."""
    cleaned_keywords = []
    for k in keywords:
        keyword_parts = re.split(r'[- ]', k.strip())
        escaped_parts = [re.escape(part) for part in keyword_parts]
        flexible_k = r'(?:-|\s)?'.join(escaped_parts)
        cleaned_keywords.append(flexible_k)
    cleaned_keywords.sort(key=len, reverse=True)
    pattern = r'\b(' + '|'.join(cleaned_keywords) + r')\b'
    cleaned_title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
    return cleaned_title.replace('-', ' ').replace('  ', ' ')

def process_image(design_img, mockup_img, mockup_config, user_config):
    """Xử lý và ghép ảnh."""
    # ... (Toàn bộ hàm này được giữ nguyên, không cần thay đổi)
    design_w, design_h = design_img.size
    pixels = design_img.load()
    visited = set()
    corner_points = [(0, 0), (design_w - 1, 0), (0, design_h - 1), (design_w - 1, design_h - 1)]
    for start_x, start_y in corner_points:
        if (start_x, start_y) in visited: continue
        seed_color = design_img.getpixel((start_x, start_y))
        seed_r, seed_g, seed_b = seed_color[:3]
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited or not (0 <= x < design_w and 0 <= y < design_h): continue
            visited.add((x, y))
            current_pixel = pixels[x, y]
            current_r, current_g, current_b = current_pixel[:3]
            if abs(current_r - seed_r) < 30 and abs(current_g - seed_g) < 30 and abs(current_b - seed_b) < 30:
                pixels[x, y] = (0, 0, 0, 0)
                stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    trimmed_design = get_trimmed_image_with_padding(design_img)
    if not trimmed_design: return None
    mockup_w, mockup_h = mockup_config['w'], mockup_config['h']
    design_w, design_h = trimmed_design.size
    scale = min(mockup_w / design_w, mockup_h / design_h)
    final_w, final_h = int(design_w * scale), int(design_h * scale)
    resized_design = trimmed_design.resize((final_w, final_h), Image.Resampling.LANCZOS)
    final_x, final_y = mockup_config['x'] + (mockup_w - final_w) // 2, mockup_config['y'] + 20
    final_mockup = mockup_img.copy()
    final_mockup.paste(resized_design, (final_x, final_y), resized_design)
    watermark_content = user_config.get("watermark_text")
    if watermark_content:
        if watermark_content.startswith(('http://', 'https://')):
            watermark_img = download_image(watermark_content)
            if watermark_img:
                max_wm_width = 280
                wm_w, wm_h = watermark_img.size
                if wm_w > max_wm_width:
                    aspect_ratio = wm_h / wm_w
                    new_w, new_h = max_wm_width, int(max_wm_width * aspect_ratio)
                    watermark_img = watermark_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                wm_w, wm_h = watermark_img.size
                paste_x, paste_y = final_mockup.width - wm_w - 20, final_mockup.height - wm_h - 50
                final_mockup.paste(watermark_img, (paste_x, paste_y), watermark_img)
        else:
            draw = ImageDraw.Draw(final_mockup)
            try:
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verdanab.ttf")
                font = ImageFont.truetype(font_path, 100)
            except IOError: font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), watermark_content, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x, text_y = final_mockup.width - text_w - 20, final_mockup.height - text_h - 50
            draw.text((text_x, text_y), watermark_content, fill=(0, 0, 0, 128), font=font)
    return final_mockup

def get_repo_size(path='.'):
    """Tính toán kích thước của repo."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def cleanup_old_zips():
    """Xóa các file .zip cũ trong thư mục output."""
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

# --- CÁC HÀM MỚI CHO PC (ĐÃ CẬP NHẬT) ---

def commit_and_push_changes_locally():
    """
    THAY ĐỔI: Thực hiện git add, commit --amend, và push --force.
    Chỉ chạy trên PC.
    """
    print("Bắt đầu quá trình commit và push...")
    try:
        os.chdir(REPO_ROOT)
        
        # Add các file đã tạo
        subprocess.run(['git', 'add', 'generate_log.txt'], check=True)

        # Kiểm tra xem có thay đổi nào không
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not result.stdout.strip():
            print("Không có thay đổi để commit.")
            return False

        print("Phát hiện thay đổi. Bắt đầu amend commit...")
        # Sử dụng --amend để gộp vào commit trước đó, --no-edit để không mở editor
        # Điều này giả định bạn đã có ít nhất một commit ban đầu với message phù hợp.
        subprocess.run(['git', 'commit', '--amend', '--no-edit'], check=True)
        
        # Lấy tên nhánh hiện tại
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
            capture_output=True, text=True, check=True
        )
        current_branch = branch_result.stdout.strip()
        
        print(f"Commit amend thành công. Bắt đầu force push lên nhánh '{current_branch}'...")
        # Phải dùng --force vì lịch sử đã bị thay đổi bởi --amend
        subprocess.run(['git', 'push', '--force', 'origin', current_branch], check=True)
        
        print("Push thành công.")
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình Git: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Lỗi: Lệnh 'git' không được tìm thấy. Hãy đảm bảo Git đã được cài đặt và có trong PATH.")
        return False

def send_telegram_log_locally():
    """
    THAY ĐỔI: Gửi nội dung log qua Telegram, đọc secrets từ .env.
    Chỉ chạy trên PC.
    """
    # Đọc token và chat_id từ biến môi trường đã được load từ file .env
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Cảnh báo: Không tìm thấy TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong file .env. Bỏ qua việc gửi log.")
        return

    log_file_path = os.path.join(REPO_ROOT, "generate_log.txt")
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        
        log_content += "\nPush successful (from PC - amended)."
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': log_content, 'parse_mode': 'HTML'}
        
        print("Đang gửi log tới Telegram...")
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        print("Gửi log tới Telegram thành công.")
    except Exception as e:
        print(f"Lỗi khi gửi nội dung log tới Telegram: {e}")


# --- (Hàm main() và các hàm còn lại giữ nguyên cấu trúc) ---
def main():
    # ...
    # (Toàn bộ logic xử lý ảnh và tạo file zip không thay đổi)
    # ...
    print("Bắt đầu quy trình tự động generate mockup.")
    output_path = os.path.join(REPO_ROOT, OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    cleanup_old_zips()
    configs = load_config()
    defaults = configs.get("defaults", {})
    output_format = defaults.get("global_output_format", "webp").lower()
    exif_defaults = defaults.get("exif_defaults", {}) 
    domains_configs = configs.get("domains", {})
    mockup_sets_config = configs.get("mockup_sets", {})
    title_clean_keywords = defaults.get("title_clean_keywords", [])
    global_skip_keywords = defaults.get("global_skip_keywords", [])

    # --- LẤY DỮ LIỆU LOG THEO MÔI TRƯỜNG ---
    log_content = ""
    try:
        if IS_GITHUB_ACTIONS:
            log_url = "https://raw.githubusercontent.com/ktbihow/imagecrawler/main/imagecrawler.log"
            log_content = requests.get(log_url).text
        else:
            with open(CRAWLER_LOG_FILE, 'r', encoding='utf-8') as f:
                log_content = f.read()
    except Exception as e:
        print(f"Lỗi: Không thể tải/đọc file imagecrawler.log. {e}")
        return

    lines = log_content.splitlines()
    domains_to_process = {}
    for line in lines:
        if "New Images" in line:
            parts = line.split(":")
            domain = parts[0].strip()
            new_urls_count = int(parts[1].split()[0])
            if new_urls_count > 0:
                domains_to_process[domain] = new_urls_count
    if not domains_to_process:
        print("Không có URL mới nào được tìm thấy. Kết thúc.")
        return
        
    urls_summary = {}
    images_for_zip = {}
    
    for domain, new_count in domains_to_process.items():
        print(f"Bắt đầu xử lý {new_count} ảnh mới từ domain: {domain}")
        
        all_urls = []
        try:
            if IS_GITHUB_ACTIONS:
                urls_url = f"https://raw.githubusercontent.com/ktbihow/imagecrawler/main/domain/{domain}.txt"
                all_urls_content = requests.get(urls_url).text
                all_urls = [line.strip() for line in all_urls_content.splitlines() if line.strip()]
            else:
                domain_file_path = os.path.join(CRAWLER_DOMAIN_DIR, f"{domain}.txt")
                with open(domain_file_path, 'r', encoding='utf-8') as f:
                    all_urls = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Lỗi: Không thể tải/đọc file URL cho domain {domain}. Bỏ qua. {e}")
            continue
        
        urls_to_process = all_urls[:new_count]
        processed_by_mockup = {}
        skipped_count = 0
        mockup_cache = {}
        mockups_to_load = set()
        domain_rules = domains_configs.get(domain, [])
        domain_rules.sort(key=lambda x: len(x.get('pattern', '')), reverse=True)

        for rule in domain_rules:
            mockup_sets_to_use = rule.get("mockup_sets_to_use", [])
            for mockup_name in mockup_sets_to_use:
                mockups_to_load.add(mockup_name)
        for mockup_name in mockups_to_load:
            if mockup_name not in mockup_sets_config:
                print(f"Lỗi: Không tìm thấy mockup set '{mockup_name}'.")
                continue
            mockup_config = mockup_sets_config.get(mockup_name)
            mockup_cache[mockup_name] = {
                "white": download_image(mockup_config.get("white")),
                "black": download_image(mockup_config.get("black")),
                "coords": mockup_config.get("coords"),
                "watermark_text": mockup_config.get("watermark_text"),
                "title_prefix_to_add": mockup_config.get("title_prefix_to_add", ""),
                "title_suffix_to_add": mockup_config.get("title_suffix_to_add", "")
            }
        for url in urls_to_process:
            if get_repo_size(REPO_ROOT) >= MAX_REPO_SIZE_MB:
                print(f"Đã đạt giới hạn dung lượng. Dừng lại.")
                break
            filename = os.path.basename(url)
            if should_globally_skip(filename, global_skip_keywords):
                skipped_count += 1
                continue
            matched_rule = next((rule for rule in domain_rules if rule.get("pattern", "") in filename), None)
            if not matched_rule or matched_rule.get("action") == "skip":
                print(f"Skipping: Rule not found or action is 'skip' for file: {filename}")
                skipped_count += 1
                continue
            mockup_sets_to_use = matched_rule.get("mockup_sets_to_use", [])
            if not mockup_sets_to_use:
                skipped_count += 1
                continue
            try:
                img = download_image(url)
                if not img:
                    skipped_count += 1
                    continue
                crop_coords = matched_rule.get("coords")
                if not crop_coords:
                    skipped_count += 1
                    continue
                pixel = img.getpixel((crop_coords['x'], crop_coords['y']))
                avg_brightness = sum(pixel[:3]) / 3
                is_white = avg_brightness > 128
                if matched_rule.get("skipWhite") and is_white:
                    skipped_count += 1
                    continue
                if matched_rule.get("skipBlack") and not is_white:
                    skipped_count += 1
                    continue
                cropped_img = img.crop((crop_coords['x'], crop_coords['y'], crop_coords['x'] + crop_coords['w'], crop_coords['y'] + crop_coords['h']))
                for mockup_name in mockup_sets_to_use:
                    if mockup_name not in mockup_cache: continue
                    mockup_data = mockup_cache.get(mockup_name)
                    mockup_to_use = mockup_data["white"] if is_white else mockup_data["black"]
                    if not mockup_to_use: continue
                    user_config = {"watermark_text": mockup_data.get("watermark_text")}
                    final_mockup = process_image(cropped_img, mockup_to_use, mockup_data.get("coords"), user_config)
                    if not final_mockup: continue
                    base_filename = os.path.splitext(filename)[0]
                    pre_clean_pattern = matched_rule.get("pre_clean_regex")
                    if pre_clean_pattern: base_filename = re.sub(pre_clean_pattern, '', base_filename)
                    cleaned_title = clean_title(base_filename.replace('-', ' ').strip(), title_clean_keywords)
                    prefix_to_add = mockup_data.get("title_prefix_to_add", "")
                    suffix_to_add = mockup_data.get("title_suffix_to_add", "")
                    final_filename_base = f"{prefix_to_add} {cleaned_title} {suffix_to_add}".replace('  ', ' ').strip()
                    save_format_pillow, file_extension = ("JPEG", ".jpg") if output_format in ["jpeg", "jpg"] else ("WEBP", ".webp")
                    image_to_save = final_mockup.convert('RGB') if save_format_pillow == "JPEG" else final_mockup
                    final_filename = final_filename_base + file_extension
                    exif_bytes = create_exif_data(prefix=mockup_name, final_filename=final_filename, exif_defaults=exif_defaults)
                    img_byte_arr = BytesIO()
                    image_to_save.save(img_byte_arr, format=save_format_pillow, quality=90, exif=exif_bytes)
                    if mockup_name not in images_for_zip: images_for_zip[mockup_name] = {}
                    if domain not in images_for_zip[mockup_name]: images_for_zip[mockup_name][domain] = []
                    images_for_zip[mockup_name][domain].append((final_filename, img_byte_arr.getvalue()))
                    processed_by_mockup[mockup_name] = processed_by_mockup.get(mockup_name, 0) + 1
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {url}: {e}")
                skipped_count += 1
        urls_summary[domain] = {'processed_by_mockup': processed_by_mockup, 'skipped': skipped_count, 'total_to_process': new_count}
        
    for mockup_name, domains_dict in images_for_zip.items():
        for domain_name, image_list in domains_dict.items():
            if not image_list: continue
            total_images_in_zip = len(image_list)
            vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            now_vietnam = datetime.now(vietnam_tz)
            domain_prefix = domain_name.split('.')[0]
            zip_filename = f"{mockup_name}.{domain_prefix}.{now_vietnam.strftime('%Y%m%d_%H%M%S')}.{total_images_in_zip}.zip"
            zip_path = os.path.join(output_path, zip_filename)
            print(f"Đang tạo file: {zip_path} với {total_images_in_zip} ảnh.")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for filename, data in image_list:
                    zf.writestr(filename, data)

    write_log(urls_summary)
    print("Hoàn thành tạo file zip và log.")

    if not IS_GITHUB_ACTIONS:
        pushed = commit_and_push_changes_locally()
        if pushed:
            send_telegram_log_locally()
    else:
        print("Đã tạo file, các bước commit, push và gửi log sẽ do GitHub Actions đảm nhiệm.")
    
    print("Kết thúc quy trình.")

def write_log(urls_summary):
    # ... (Hàm này giữ nguyên)
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now_vietnam = datetime.now(vietnam_tz)
    log_file_path = os.path.join(REPO_ROOT, "generate_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- Summary of Last Generation ---\n")
        f.write(f"Timestamp: {now_vietnam.strftime('%Y-%m-%d %H:%M:%S')} +07\n\n")
        if not urls_summary:
            f.write("No new images were processed in this run.\n")
        else:
            for domain, counts in urls_summary.items():
                f.write(f"Domain: {domain}\n")
                processed_by_mockup = counts.get('processed_by_mockup')
                if processed_by_mockup:
                    for mockup, count in processed_by_mockup.items():
                        f.write(f"  {mockup}: {count}\n")
                f.write(f"  Skipped Images: {counts['skipped']}\n")
                f.write(f"  Total URLs to Process: {counts['total_to_process']}\n\n")
    print(f"Generation summary saved to {log_file_path}")

if __name__ == "__main__":
    main()