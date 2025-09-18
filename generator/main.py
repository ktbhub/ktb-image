import os
import requests
import json
import zipfile
import re
from datetime import datetime
import pytz
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import piexif # <--- THÊM MỚI

#Đây là bản update test thử chức năng magicwand đa điểm, nếu không work có thể back lại version backup lưu ở PCHome 10:33PM 9.10.25
# --- Cấu hình ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = "generated-zips"
CONFIG_FILE = os.path.join(REPO_ROOT, "generator", "config.json")

# Giới hạn dung lượng repo GitHub
MAX_REPO_SIZE_MB = 900

# --- Các hàm hỗ trợ ---

def create_exif_data(prefix, final_filename):
    """
    Tạo chuỗi bytes EXIF dựa trên prefix và tên file cuối cùng.
    Sử dụng các thẻ EXIF tương đương để đáp ứng yêu cầu.
    """
    domain = prefix + ".com"
    now_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    software_name = b"Adobe Photoshop 25.0" # Tên phần mềm tùy chỉnh

    try:
        # Dữ liệu cho IFD (Image File Directory) chính - "0th"
        zeroth_ifd = {
            piexif.ImageIFD.Artist: domain.encode('utf-8'),
            piexif.ImageIFD.Copyright: domain.encode('utf-8'),
            piexif.ImageIFD.ImageDescription: final_filename.encode('utf-8'),
            piexif.ImageIFD.Software: software_name,
            # Các thẻ Windows XP yêu cầu mã hóa utf-16le
            piexif.ImageIFD.XPAuthor: domain.encode('utf-16le'),
            piexif.ImageIFD.XPComment: final_filename.encode('utf-16le'),
            piexif.ImageIFD.XPSubject: final_filename.encode('utf-16le'),
            # Từ khóa phải được ngăn cách bằng dấu ';' và kết thúc bằng null character (\x00\x00)
            piexif.ImageIFD.XPKeywords: (prefix + ";" + "shirt;").encode('utf-16le')
        }
        
        # Dữ liệu cho Exif IFD
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: now_str.encode('utf-8'),
            piexif.ExifIFD.CreateDate: now_str.encode('utf-8'),
        }

        exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd}
        exif_bytes = piexif.dump(exif_dict)
        return exif_bytes
    except Exception as e:
        print(f"Lỗi khi tạo dữ liệu EXIF: {e}")
        return b'' # Trả về bytes rỗng nếu có lỗi

def should_globally_skip(filename, skip_keywords):
    """
    Kiểm tra xem tên tệp có chứa bất kỳ TỪ KHÓA NGUYÊN VẸN nào 
    trong danh sách bỏ qua toàn cục hay không.
    Hàm này không phân biệt chữ hoa/thường và chỉ khớp với từ riêng lẻ.
    """
    for keyword in skip_keywords:
        # Tạo một biểu thức chính quy (regex) để tìm từ khóa như một từ độc lập.
        # \b là "word boundary" (ranh giới của một từ), đảm bảo nó không phải là một phần của từ khác.
        # re.escape để xử lý các ký tự đặc biệt nếu có trong keyword (ví dụ: '2-Sided').
        pattern = r'\b' + re.escape(keyword) + r'\b'
        
        # re.search tìm kiếm pattern ở bất kỳ đâu trong tên tệp, không phân biệt hoa/thường.
        if re.search(pattern, filename, re.IGNORECASE):
            # In ra log để biết chính xác từ khóa nào đã gây ra việc skip
            print(f"Skipping (Global): '{filename}' chứa từ khóa bị cấm '{keyword}'.")
            return True
            
    return False

def get_trimmed_image_with_padding(image, max_padding_x=40, max_padding_y=20):
    """Trims transparent borders but keeps a specified maximum padding."""
    bbox = image.getbbox()
    if not bbox:
        return None

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
    """Tải ảnh từ URL và trả về đối tượng PIL Image."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': url
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        print(f"Lỗi khi tải ảnh từ {url}: {e}")
        return None

def clean_title(title, keywords):
    """Làm sạch tiêu đề bằng cách loại bỏ các từ khóa không phân biệt chữ hoa/thường."""
    
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
    """Cắt, trim và dán design vào mockup."""
    
    # --- LOGIC XÓA NỀN "MAGIC WAND" 4 GÓC ---
    design_w, design_h = design_img.size
    pixels = design_img.load()
    visited = set() # Dùng chung cho tất cả các lần flood fill

    # 1. Xác định 4 điểm góc để bắt đầu
    corner_points = [
        (0, 0),                  # Góc trên-trái
        (design_w - 1, 0),       # Góc trên-phải
        (0, design_h - 1),       # Góc dưới-trái
        (design_w - 1, design_h - 1) # Góc dưới-phải
    ]

    # 2. Lặp qua từng góc và chạy một "magic wand" độc lập
    for start_x, start_y in corner_points:
        # Nếu điểm này đã được xử lý trong một lần loang màu trước đó thì bỏ qua
        if (start_x, start_y) in visited:
            continue

        # Lấy màu tham chiếu TẠI CHÍNH GÓC ĐÓ
        seed_color = design_img.getpixel((start_x, start_y))
        seed_r, seed_g, seed_b = seed_color[:3]
        
        # Bắt đầu quy trình loang màu mới từ góc này
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x, y) in visited or not (0 <= x < design_w and 0 <= y < design_h):
                continue
            
            # Đánh dấu pixel này đã được ghé thăm
            visited.add((x, y))
            
            current_pixel = pixels[x, y]
            current_r, current_g, current_b = current_pixel[:3]
            
            # So sánh màu hiện tại với màu tham chiếu của GÓC NÀY
            if abs(current_r - seed_r) < 30 and abs(current_g - seed_g) < 30 and abs(current_b - seed_b) < 30:
                pixels[x, y] = (0, 0, 0, 0) # Chuyển thành trong suốt
                
                # Thêm các điểm lân cận vào stack để xử lý tiếp
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

    # --- KẾT THÚC LOGIC XÓA NỀN NÂNG CẤP ---
            
    trimmed_design = get_trimmed_image_with_padding(design_img)
    if not trimmed_design:
        return None
    
    # Các bước còn lại giữ nguyên...
    mockup_w, mockup_h = mockup_config['w'], mockup_config['h']
    design_w, design_h = trimmed_design.size
    
    scale = min(mockup_w / design_w, mockup_h / design_h)
    
    final_w = int(design_w * scale)
    final_h = int(design_h * scale)
    
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
                max_wm_width = 280
                wm_w, wm_h = watermark_img.size
                if wm_w > max_wm_width:
                    aspect_ratio = wm_h / wm_w
                    new_w = max_wm_width
                    new_h = int(new_w * aspect_ratio)
                    watermark_img = watermark_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                wm_w, wm_h = watermark_img.size
                paste_x = final_mockup.width - wm_w - 20
                paste_y = final_mockup.height - wm_h - 50
                final_mockup.paste(watermark_img, (paste_x, paste_y), watermark_img)
        else:
            draw = ImageDraw.Draw(final_mockup)
            try:
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verdanab.ttf")
                font = ImageFont.truetype(font_path, 100)
            except IOError:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), watermark_content, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_x = final_mockup.width - text_w - 20
            text_y = final_mockup.height - text_h - 50
            draw.text((text_x, text_y), watermark_content, fill=(0, 0, 0, 128), font=font)
    
    return final_mockup

def get_repo_size(path='.'):
    """Tính toán kích thước của repo hiện tại."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def cleanup_old_zips():
    """Xóa TOÀN BỘ file .zip trong thư mục output khi action bắt đầu."""
    output_path = os.path.join(REPO_ROOT, OUTPUT_DIR)
    if not os.path.exists(output_path):
        return

    print("Bắt đầu dọn dẹp tất cả các file zip cũ...")
    
    for filename in os.listdir(output_path):
        if filename.endswith(".zip"):
            file_path = os.path.join(output_path, filename)
            try:
                os.remove(file_path)
                print(f"Đã xóa: {filename}")
            except Exception as e:
                print(f"Lỗi khi xóa file {filename}: {e}")
                
    print("Dọn dẹp hoàn tất.")

# --- Logic chính ---

def main():
    print("Bắt đầu quy trình tự động generate mockup.")

    output_path = os.path.join(REPO_ROOT, OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cleanup_old_zips()

    configs = load_config()
    defaults = configs.get("defaults", {})
    domains_configs = configs.get("domains", {})
    mockup_sets_config = configs.get("mockup_sets", {})

    title_clean_keywords = defaults.get("title_clean_keywords", [])
    global_skip_keywords = defaults.get("global_skip_keywords", [])

    try:
        log_url = "https://raw.githubusercontent.com/ktbihow/imagecrawler/main/imagecrawler.log"
        log_content = requests.get(log_url).text
    except Exception as e:
        print(f"Lỗi: Không thể tải file imagecrawler.log từ GitHub. {e}")
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

        current_size = get_repo_size(REPO_ROOT)
        if current_size >= MAX_REPO_SIZE_MB:
            print(f"Cảnh báo: Dung lượng repo đã vượt quá {MAX_REPO_SIZE_MB}MB. Bỏ qua.")
            break

        domain_rules = domains_configs.get(domain, [])

        if not domain_rules:
            print(f"Không có quy tắc nào được định nghĩa cho domain {domain}. Bỏ qua.")
            continue

        domain_rules.sort(key=lambda x: len(x.get('pattern', '')), reverse=True)

        try:
            urls_url = f"https://raw.githubusercontent.com/ktbihow/imagecrawler/main/domain/{domain}.txt"
            all_urls_content = requests.get(urls_url).text
            all_urls = [line.strip() for line in all_urls_content.splitlines() if line.strip()]
        except Exception as e:
            print(f"Lỗi: Không thể tải file URL cho domain {domain}. Bỏ qua. {e}")
            continue

        urls_to_process = all_urls[:new_count]

        processed_by_mockup = {}
        skipped_count = 0

        mockup_cache = {}
        mockups_to_load = set()
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

            # >> KIỂM TRA BỎ QUA TOÀN CỤC NGAY TẠI ĐÂY <<
            if should_globally_skip(filename, global_skip_keywords):
                # Message is already printed inside the function
                skipped_count += 1
                continue

            matched_rule = next((rule for rule in domain_rules if rule.get("pattern", "") in filename), None)

            if not matched_rule or matched_rule.get("action") == "skip":
                print(f"Skipping: Rule not found or action is 'skip' for file: {filename}")
                skipped_count += 1
                continue

            mockup_sets_to_use = matched_rule.get("mockup_sets_to_use", [])
            if not mockup_sets_to_use:
                print(f"Không có mockup sets nào được chỉ định cho domain {domain}. Bỏ qua.")
                skipped_count += 1
                continue

            try:
                img = download_image(url)
                if not img:
                    skipped_count += 1
                    continue

                crop_coords = matched_rule.get("coords")
                if not crop_coords:
                    print(f"Không có tọa độ crop trong rule cho file {filename}. Bỏ qua.")
                    skipped_count += 1
                    continue

                pixel = img.getpixel((crop_coords['x'], crop_coords['y']))
                avg_brightness = sum(pixel[:3]) / 3
                is_white = avg_brightness > 128

                if matched_rule.get("skipWhite") and is_white:
                    print(f"Skipping white shirt as per rule for file: {filename}")
                    skipped_count += 1
                    continue

                if matched_rule.get("skipBlack") and not is_white:
                    print(f"Skipping black shirt as per rule for file: {filename}")
                    skipped_count += 1
                    continue

                cropped_img = img.crop((crop_coords['x'], crop_coords['y'], crop_coords['x'] + crop_coords['w'], crop_coords['y'] + crop_coords['h']))

                for mockup_name in mockup_sets_to_use:
                    if mockup_name not in mockup_cache:
                        continue

                    mockup_data = mockup_cache.get(mockup_name)
                    mockup_to_use = mockup_data["white"] if is_white else mockup_data["black"]

                    if not mockup_to_use:
                        print(f"Lỗi khi tải mockup {mockup_name}. Bỏ qua.")
                        continue

                    user_config = {
                        "watermark_text": mockup_data.get("watermark_text")
                    }

                    final_mockup = process_image(cropped_img, mockup_to_use, mockup_data.get("coords"), user_config)
                    if not final_mockup:
                        continue

                    base_filename = os.path.splitext(filename)[0]
                    # --- BẮT ĐẦU THAY ĐỔI ---
                    # Áp dụng regex làm sạch sơ bộ nếu được định nghĩa trong rule
                    pre_clean_pattern = matched_rule.get("pre_clean_regex")
                    if pre_clean_pattern:
                        base_filename = re.sub(pre_clean_pattern, '', base_filename)
                    # --- KẾT THÚC THAY ĐỔI ---
                    cleaned_title = clean_title(base_filename.replace('-', ' ').strip(), title_clean_keywords)

                    prefix_to_add = mockup_data.get("title_prefix_to_add", "")
                    suffix_to_add = mockup_data.get("title_suffix_to_add", "")

                    final_filename = f"{prefix_to_add} {cleaned_title} {suffix_to_add}".replace('  ', ' ').strip()
                    final_filename += '.webp'
                    
                    # --- BẮT ĐẦU TÍCH HỢP EXIF ---
                    # Lấy prefix từ domain, ví dụ 'teepublic' từ 'teepublic.com'
                    prefix = domain.split('.')[0]
                    
                    # Tạo dữ liệu EXIF
                    exif_bytes = create_exif_data(prefix, final_filename)
                    
                    # Lưu ảnh với dữ liệu EXIF
                    img_byte_arr = BytesIO()
                    final_mockup.save(img_byte_arr, format="WEBP", quality=90, exif=exif_bytes)
                    # --- KẾT THÚC TÍCH HỢP EXIF ---

                    if mockup_name not in images_for_zip:
                        images_for_zip[mockup_name] = []
                    
                    images_for_zip[mockup_name].append((final_filename, img_byte_arr.getvalue()))
                    processed_by_mockup[mockup_name] = processed_by_mockup.get(mockup_name, 0) + 1

            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {url}: {e}")
                skipped_count += 1
        
        urls_summary[domain] = {
            'processed_by_mockup': processed_by_mockup,
            'skipped': skipped_count,
            'total_to_process': new_count
        }

    for mockup_name, image_list in images_for_zip.items():
        if not image_list:
            continue
            
        total_images_in_zip = len(image_list)
        
        # Lấy múi giờ Việt Nam
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now_vietnam = datetime.now(vietnam_tz)
        
        zip_filename = f"{mockup_name}.{now_vietnam.strftime('%Y%m%d_%H%M%S')}_{total_images_in_zip}_images.zip"
        
        zip_path = os.path.join(output_path, zip_filename)
        
        print(f"Đang tạo file: {zip_path} với {total_images_in_zip} ảnh.")
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for filename, data in image_list:
                zf.writestr(filename, data)

    write_log(urls_summary)
    print("Kết thúc quy trình.")

def write_log(urls_summary):
    """Ghi tóm tắt kết quả generate vào file generate_log.txt."""
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now_vietnam = datetime.now(vietnam_tz)
    
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generate_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- Summary of Last Generation ---\n")
        # Định dạng timestamp với múi giờ +07
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
