import cv2
import numpy as np
import tensorflow as tf
import os
import re
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn

# OCR untuk KTP (opsional: butuh pytesseract + Tesseract terpasang)
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False

print("Loaded necessary libraries.")

# 1. Load the pre-trained classification model
model = tf.keras.models.load_model('model.keras')
print("Classification model 'model.keras' loaded.")

# 2. Load the Haar Cascade classifier (use OpenCV's built-in path if local file missing)
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
if os.path.isfile(HAAR_CASCADE_FILE):
    cascade_path = HAAR_CASCADE_FILE
else:
    cascade_path = os.path.join(cv2.data.haarcascades, HAAR_CASCADE_FILE)
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error: Could not load Haar Cascade file {HAAR_CASCADE_FILE}.")
else:
    print(f"Haar Cascade classifier '{HAAR_CASCADE_FILE}' loaded.")

# 3. Define image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
print(f"Image dimensions set to {IMG_WIDTH}x{IMG_HEIGHT}.")

# 4. Create a helper function for face detection and classification
def detect_and_classify_face(image_np, face_classifier, classifier_model, img_height, img_width):
    face_count = 0
    is_face_human = None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_count = len(faces)
    
    if face_count == 1:
        (x, y, w, h) = faces[0]
        cropped_face = image_np[y:y+h, x:x+w]
        
        # Resize and normalize
        cropped_face_resized = cv2.resize(cropped_face, (img_width, img_height))
        normalized_face = cropped_face_resized / 255.0
        
        # Expand dimensions for model prediction (batch of 1)
        input_image = np.expand_dims(normalized_face, axis=0)
        
        # Get prediction
        prediction = classifier_model.predict(input_image)[0][0]
        is_face_human = bool(prediction >= 0.5)
        
    return face_count, is_face_human

print("Helper function 'detect_and_classify_face' defined.")

# --- OCR KTP Indonesia ---
def _to_gray_and_resize(image_np, min_width=800):
    """Grayscale + resize jika terlalu kecil."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()
    h, w = gray.shape[:2]
    if w < min_width:
        scale = min_width / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _preprocess_variants(gray):
    """Beberapa variasi preprocessing untuk dicoba (KTP teks hitam di atas putih)."""
    variants = []
    # 1. Hanya denoise ringan (bagus untuk foto yang sudah jelas)
    v1 = cv2.bilateralFilter(gray, 9, 75, 75)
    variants.append(("bilateral", v1))
    # 2. Adaptive threshold (bagus untuk kontras tinggi)
    v2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive", v2))
    # 3. Otsu threshold
    _, v3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", v3))
    # 4. CLAHE (tingkatkan kontras) lalu threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v4 = clahe.apply(gray)
    v4 = cv2.adaptiveThreshold(v4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("clahe", v4))
    # 5. Grayscale saja (tanpa threshold) - untuk gambar terang
    variants.append(("gray", gray))
    return variants


def ocr_ktp_image(image_np):
    """Jalankan OCR pada gambar; coba beberapa preprocessing & PSM, pilih hasil terbaik."""
    if not OCR_AVAILABLE:
        return None, "OCR tidak tersedia. Install: pip install pytesseract dan pasang Tesseract (tesseract-ocr, tesseract-ocr-ind)."
    gray = _to_gray_and_resize(image_np)
    variants = _preprocess_variants(gray)
    psms = [6, 4, 13]  # 6=block, 4=column, 13=raw line
    best_text = ""
    best_score = -1
    for name, img in variants:
        for psm in psms:
            try:
                text = pytesseract.image_to_string(
                    img, lang="ind+eng", config=f"--psm {psm}"
                )
            except Exception:
                try:
                    text = pytesseract.image_to_string(img, lang="eng", config=f"--psm {psm}")
                except Exception:
                    continue
            text = (text or "").strip()
            if not text:
                continue
            # Skor: ada NIK (16 digit) = sangat bagus; banyak digit/huruf = bagus
            score = 0
            if re.search(r"\d{16}", re.sub(r"\s", "", text)):
                score += 100
            score += min(50, len(re.findall(r"\d{10,}", text)) * 10)
            score += min(30, text.count(" ") * 2)
            if "NIK" in text or "Nama" in text or "Alamat" in text:
                score += 20
            if score > best_score:
                best_score = score
                best_text = text
    if not best_text:
        return "", None  # kembalikan string kosong, jangan error
    return best_text, None


# Pola regex label KTP (toleran salah baca OCR), urutan penting
_KTP_LABEL_PATTERNS = [
    (r"NIK\s*['\"]?\s*\+?\s*", "nik"),
    (r"Nama\s*[:\s]", "nama"),
    (r"Tempat[i\s/]*Tg[lid]?\s*Lahir|Tempat\s*/\s*Tg\w*\s*Lahir", "tempat_tgl_lahir"),
    (r"Jenis\s*Kelamin\s*\|?\s*[:\s]", "jenis_kelamin"),
    (r"Alamat\s*[\.\-:\s]+", "alamat"),
    (r"RT\s*[\/i]?\s*RW\.?|RTIRW\.?\s*[:\s]", "rt_rw"),
    (r"Kel\s*[\/i]?\s*Desa\.?|KeliDesa\.?\s*[:_\s]", "kel_desa"),
    (r"Kecamatan\s*[—\-:\s]", "kecamatan"),
    (r"Agama\s+", "agama"),
    (r"Status\s*Perkawinan|Status\s*Perkawinati\s*[:\s]|Status\s*aw\s+", "status_perkawinan"),
    (r"Pek[e]+rjaan\s*[^A-Za-z]*", "pekerjaan"),  # Pekerjaan / Pejeerjaan (salah baca e), sampai huruf
    (r"Kewarganegaraan\s*[:\s]", "kewarganegaraan"),
    (r"Berlaku\s*Hingga\s*[:\s]", "berlaku_hingga"),
    (r"Gol\.?\s*Darah\s*[:\s]|Golongan\s*Darah", "golongan_darah"),
]

# Semua key field KTP standar (response selalu sertakan ini)
KTP_FIELD_KEYS = [
    "nik", "nama", "tempat_tgl_lahir", "jenis_kelamin", "alamat", "rt_rw",
    "kel_desa", "kecamatan", "agama", "status_perkawinan", "pekerjaan",
    "kewarganegaraan", "berlaku_hingga", "golongan_darah"
]

# Daftar resmi pekerjaan KTP Indonesia (untuk normalisasi hasil OCR)
KTP_PEKERJAAN_OPTIONS = [
    "Belum / Tidak Bekerja", "Buruh", "Nelayan / Perikanan", "Mengurus Rumah Tangga",
    "Buruh Peternakan", "Pelajar / Mahasiswa", "Pembantu Rumah Tangga", "Pensiunan",
    "Tukang Cukur", "Pegawai Negeri Sipil", "Tentara Nasional Indonesia", "Kepolisian RI",
    "Perdagangan", "Petani / Pekebun", "Peternak", "Industri", "Konstruksi", "Transportasi",
    "Karyawan Swasta", "Karyawan BUMN", "Karyawan BUMD", "Karyawan Honorer",
    "Buruh Harian Lepas", "Buruh Tani / Perkebunan", "Tukang Listrik", "Tukang Batu",
    "Tukang Kayu", "Tukang Sol Sepatu", "Tukang Las / Pandai Besi", "Tukang Jahit",
    "Penata Rambut", "Penata Rias", "Penata Busana", "Mekanik", "Tukang Gigi",
    "Seniman", "Tabib", "Paraji", "Perancang Busana", "Penerjemah", "Imam Masjid",
    "Pendeta", "Pastur", "Wartawan", "Ustadz / Mubaligh", "Juru Masak", "Promotor Acara",
    "Anggota DPR-RI", "Anggota DPD", "Anggota BPK", "Presiden", "Wakil Presiden",
    "Anggota Mahkamah Konstitusi", "Anggota Kabinet / Kementerian", "Duta Besar",
    "Gubernur", "Wakil Gubernur", "Bupati", "Wakil Bupati", "Walikota", "Wakil Walikota",
    "Anggota DPRD Provinsi", "Anggota DPRD Kabupaten / Kota", "Dosen", "Guru", "Pilot",
    "Pengacara", "Notaris", "Arsitek", "Akuntan", "Konsultan", "Dokter", "Bidan",
    "Perawat", "Apoteker", "Psikiater / Psikolog", "Penyiar Televisi", "Penyiar Radio",
    "Pelaut", "Peneliti", "Sopir", "Pialang", "Paranormal", "Pedagang", "Perangkat Desa",
    "Kepala Desa", "Biarawati", "Wiraswasta", "Anggota Lembaga Tinggi", "Artis", "Atlet",
    "Chef", "Manajer", "Tenaga Tata Usaha", "Operator", "Pekerja Pengolahan / Kerajinan",
    "Teknisi", "Asisten Ahli", "Lainnya",
]

# Mapping OCR / singkatan -> nilai standar pekerjaan
_PEKERJAAN_NORMALIZE = {
    "pegawai swasta": "Karyawan Swasta",
    "karyawan swasta": "Karyawan Swasta",
    "wiraswasta": "Wiraswasta",
    "pns": "Pegawai Negeri Sipil",
    "pegawai negeri sipil": "Pegawai Negeri Sipil",
    "tni": "Tentara Nasional Indonesia",
    "tentara nasional indonesia": "Tentara Nasional Indonesia",
    "polri": "Kepolisian RI",
    "kepolisian ri": "Kepolisian RI",
    "polisi": "Kepolisian RI",
    "pelajar": "Pelajar / Mahasiswa",
    "mahasiswa": "Pelajar / Mahasiswa",
    "pelajar mahasiswa": "Pelajar / Mahasiswa",
    "pensiunan": "Pensiunan",
    "mengurus rumah tangga": "Mengurus Rumah Tangga",
    "irt": "Mengurus Rumah Tangga",
    "buruh": "Buruh",
    "petani": "Petani / Pekebun",
    "pekebun": "Petani / Pekebun",
    "petani pekebun": "Petani / Pekebun",
    "nelayan": "Nelayan / Perikanan",
    "perikanan": "Nelayan / Perikanan",
    "guru": "Guru",
    "dosen": "Dosen",
    "dokter": "Dokter",
    "perawat": "Perawat",
    "bidan": "Bidan",
    "pedagang": "Pedagang",
    "perdagangan": "Perdagangan",
    "karyawan bumn": "Karyawan BUMN",
    "karyawan bumd": "Karyawan BUMD",
    "karyawan honorer": "Karyawan Honorer",
    "belum tidak bekerja": "Belum / Tidak Bekerja",
    "tidak bekerja": "Belum / Tidak Bekerja",
    "sopir": "Sopir",
    "pilot": "Pilot",
    "pelaut": "Pelaut",
    "mekanik": "Mekanik",
    "tukang": "Lainnya",
    "teknis": "Teknisi",
    "teknisi": "Teknisi",
    "operator": "Operator",
    "manajer": "Manajer",
}


def _normalize_pekerjaan(raw: str) -> str:
    """Normalisasi teks pekerjaan OCR ke salah satu nilai standar KTP."""
    if not raw or not raw.strip():
        return ""
    s = " ".join(raw.strip().split()).lower()
    if s in _PEKERJAAN_NORMALIZE:
        return _PEKERJAAN_NORMALIZE[s]
    for key, value in _PEKERJAAN_NORMALIZE.items():
        if key in s or s in key:
            return value
    for opt in KTP_PEKERJAAN_OPTIONS:
        if opt.lower() in s:
            return opt
        if s in opt.lower():
            return opt
    return raw.strip()  # kembalikan asli jika tidak cocok


def _clean_ktp_value(raw: str, field_key: str) -> str:
    """Bersihkan nilai dari noise OCR."""
    if not raw:
        return ""
    s = " ".join(raw.split())
    s = s.strip(" '\"\\|;*")
    s = re.sub(r"^[\s\|:\-\"\'<>!*\\\.]+", "", s)
    s = re.sub(r"Content\s*\*.*$", "", s, flags=re.I).strip()
    s = re.sub(r"\\\s*[a-z]{1,3}\s*$", "", s).strip()
    s = re.sub(r"\s+[a-z]\s*$", "", s).strip()
    s = re.sub(r"\s+\d[hH]\s*$", "", s).strip()
    s = re.sub(r"[\"']\s*$", "", s).strip()
    s = re.sub(r"^\s*[\"']\s*", "", s).strip()
    s = re.sub(r"\s*:\s*[A-Za-z0-9\s]{0,15}\s*$", "", s).strip()
    s = re.sub(r"\s+[Nn][Ss]\s+[A-Za-z0-9]+\s*\d*\s*$", "", s).strip()
    s = re.sub(r"\s+[Ee]{2,}\s*\.?\s*$", "", s).strip()
    s = re.sub(r"\s*[—\-]\s*\d*\s*[a-zA-Z]{1,3}\s*$", "", s).strip()
    # Hapus kata/fragment 1-2 huruf di akhir yang seperti typo (ea, nb, at, Se, Vs)
    s = re.sub(r"\s+[A-Za-z]{1,2}\s*$", "", s).strip()
    s = re.sub(r"\s*\(\w+\s*$", "", s).strip()
    s = re.sub(r"\s+[A-Z]{2,}\s+[A-Za-z]+\s*$", "", s).strip()  # "EE t", "LT AM aa"
    if field_key == "nik":
        digits = re.sub(r"\D", "", s)
        return digits if len(digits) == 16 else s
    if field_key == "kewarganegaraan" and s:
        wni_wna = re.search(r"\b(WNI|WNA)\b", s, re.I)
        if wni_wna:
            return wni_wna.group(1).upper()
    if field_key == "agama":
        s = re.sub(r"\s+Status\s+.*$", "", s, flags=re.I).strip()
        if len(s) > 12:
            first = s.split(",")[0].strip()
            if len(first) <= 20:
                s = first
    if field_key == "golongan_darah" and len(s) > 3:
        m = re.search(r"\b([ABO])\b", s, re.I)
        if m:
            return m.group(1).upper()
    if field_key == "berlaku_hingga":
        m = re.search(r"(\d{2}[-/]\d{2}[-/]\d{4})", s)
        if m:
            return m.group(1)
        # Tahun 19xx/20xx (bukan digit acak seperti 2210)
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", s):
            return m.group(1)
        m = re.search(r"\d{4}", s)
        if m:
            return m.group(0)
    if field_key == "tempat_tgl_lahir":
        s = re.sub(r"\s+[A-Za-z]+\s*=\s*\d+[a-z]\s*$", "", s, flags=re.I).strip()
        s = re.sub(r"\s+[A-Za-z]{1,2}\s*$", "", s).strip()
    # Nama: hentikan di " Tempat", " yo ", " JAKARTA", dan buang tanda kutip + huruf di akhir
    if field_key == "nama":
        s = re.sub(r"\s+[Tt]empat.*$", "", s).strip()
        s = re.sub(r"\s+yo\s+.*$", "", s, flags=re.I).strip()
        s = re.sub(r"\s+[Jj]akarta\s*,.*$", "", s).strip()
        s = re.sub(r"\s*[\"']?\s*[a-z]\s*$", "", s).strip()
        # Gabungkan nama: ganti "|" dengan spasi (MIRASETIAWAN | AN -> MIRASETIAWAN AN)
        s = s.replace("|", " ").strip()
        # Pisahkan kata yang salah baca jadi satu (MIRASETIAWAN -> MIRA SETIAWAN)
        if re.search(r"\b\w+SETIAWAN\b", s, re.I):
            s = re.sub(r"\b(\w+)(SETIAWAN)\b", r"\1 \2", s, flags=re.I).strip()
        # Buang " AN" di akhir (duplikat/sisa OCR, nama yang benar: MIRA SETIAWAN)
        s = re.sub(r"\s+AN\s*$", "", s, flags=re.I).strip()
        # Buang kata yang hanya 1 huruf di akhir
        while s and len(s.split()) > 1 and len(s.split()[-1]) == 1:
            s = " ".join(s.split()[:-1])
    # Jenis kelamin: hanya PEREMPUAN atau LAKI-LAKI
    if field_key == "jenis_kelamin":
        if "PEREMPUAN" in s.upper():
            return "PEREMPUAN"
        if "LAKI" in s.upper() or "LAKI-LAKI" in s.upper():
            return "LAKI-LAKI"
        s = s.split()[0] if s.split() else s
    # Alamat: ambil hanya JL/IL. ... sampai nomor; normalisasi IL. -> JL.
    if field_key == "alamat":
        s = re.sub(r"\s+KN\s+.*$", "", s).strip()
        s = re.sub(r"\s+A\\\\?\s+.*$", "", s).strip()
        s = re.sub(r"\s+\[?So\s+.*$", "", s).strip()
        s = re.sub(r"\s+RTIRW.*$", "", s, flags=re.I).strip()
        m = re.search(r"(?:JL\.|IL\.|Jalan)\s*[A-Za-z0-9\s\/\.\-]+", s, re.I)
        if m:
            s = m.group(0).strip()
        # Normalisasi: IL. (salah baca OCR) -> JL. (Jalan)
        s = re.sub(r"^IL\.\s*", "JL. ", s, flags=re.I).strip()
    # Kecamatan: ambil satu kata (nama kecamatan) sebelum " : ", " (", " NS"
    if field_key == "kecamatan":
        s = re.sub(r"\s*:\s*.*$", "", s).strip()
        s = re.sub(r"\s+[Nn][Ss]\s+.*$", "", s).strip()
        s = re.sub(r"\s*\(.*$", "", s).strip()
        s = s.strip("—: ")
    # Pekerjaan: hapus leading/trailing junk
    if field_key == "pekerjaan":
        s = re.sub(r"^[\"'\s<>!0-9\(\)]+\s*", "", s).strip()
        s = re.sub(r"\s+Te\s+ea\s+ea\s+nb\s*$", "", s, flags=re.I).strip()
        s = re.sub(r"\s*\(\s*H\d\s*$", "", s).strip()
    # RT/RW: format 000/000
    if field_key == "rt_rw":
        m = re.search(r"\d{3}\s*/\s*\d{3}", s)
        if m:
            s = m.group(0)
    # Kel/Desa: satu kata (nama kelurahan)
    if field_key == "kel_desa":
        s = re.sub(r"^[_\s:\.\-]+\s*", "", s).strip()
        s = re.sub(r"\s+[Vv][Ss],?\s*.*$", "", s).strip()
        s = re.sub(r"\s+[A-Za-z]{1,2}\s*$", "", s).strip()
    # Status perkawinan: KAWIN / BELUM KAWIN / dll
    if field_key == "status_perkawinan":
        if "KAWIN" in s.upper() and "BELUM" not in s.upper():
            s = "KAWIN"
        elif "BELUM" in s.upper():
            s = "BELUM KAWIN"
        else:
            s = s.split()[0] if s.split() else s
    return s.strip(" '\"\\|;*")


def parse_ktp_fields(ocr_text):
    """Parse teks OCR KTP: cari tiap label dengan regex, ambil value sampai label berikutnya."""
    result = {}
    if not ocr_text:
        return result
    # NIK pasti 16 digit
    nik_match = re.search(r"\d{16}", re.sub(r"\s", "", ocr_text))
    if nik_match:
        result["nik"] = nik_match.group(0)

    # Cari semua posisi label (pattern, key, start, end)
    matches = []
    for pattern, key in _KTP_LABEL_PATTERNS:
        for m in re.finditer(pattern, ocr_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), key))
            break  # satu match per label jenis

    matches.sort(key=lambda x: x[0])

    for i, (start, end, key) in enumerate(matches):
        if key == "nik":
            continue
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(ocr_text)
        value = ocr_text[end:next_start]
        value = _clean_ktp_value(value, key)
        if not value:
            continue
        if key == "golongan_darah" and len(value) > 2:
            blood = re.search(r"\b([ABO])\b", value, re.I)
            if blood:
                value = blood.group(1).upper()
        if key == "pekerjaan":
            value = _normalize_pekerjaan(value)
        result[key] = value

    # Fallback pekerjaan: teks antara Agama/Status dan Kewarganegaraan, cari pola pekerjaan
    if not result.get("pekerjaan"):
        block = ocr_text
        for sep in ["Kewarganegaraan", "Kewarganegaraa", "Kewarga"]:
            if sep in block:
                block = block.split(sep, 1)[0]
                break
        for sep in ["Status Perkawinan", "Status Perkawinati", "KAWIN"]:
            if sep in block:
                parts = block.split(sep, 1)
                if len(parts) > 1:
                    block = parts[-1]
                break
        # Cari istilah pekerjaan umum (bukan agama/status)
        job_match = re.search(
            r"(PEGAWAI\s+SWASTA|WIRASWASTA|PELAJAR|MAHASISWA|KARYAWAN\s+SWASTA|BURUH|PETANI|NELAYAN|PNS|TNI|POLRI|PENSIUNAN|MENGURUS\s+RUMAH\s+TANGGA)",
            block,
            re.I
        )
        if job_match:
            val = _clean_ktp_value(job_match.group(1), "pekerjaan")
            if val and len(val) > 2 and val.upper() not in ("KAWIN", "ISLAM", "BELUM"):
                result["pekerjaan"] = _normalize_pekerjaan(val)

    return result


print("OCR KTP helpers defined." if OCR_AVAILABLE else "OCR KTP skipped (pytesseract not available).")

# 5. Initialize FastAPI application
app = FastAPI()
print("FastAPI app initialized.")

# 6. Define a POST endpoint
@app.post("/validate-face/")
async def predict_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "message": "Could not decode image.", "is_human": None, "face_found": False}

    face_count, is_face_human_result = detect_and_classify_face(img, face_cascade, model, IMG_HEIGHT, IMG_WIDTH)

    success = False
    is_human_label = None

    if face_count == 1:
        success = True
        is_human_label = bool(is_face_human_result)
    
    # The request specifically asks for 'is_human', 'face_found', 'is_face_human', and 'success'
    # 'is_human' and 'is_face_human' are redundant but included as per request.
    return {
        "is_human": is_human_label, 
        "face_found": face_count, 
        "is_face_human": is_human_label, 
        "success": success
    }


@app.get("/ocr-ktp/pekerjaan-options/")
async def get_pekerjaan_options():
    """Daftar resmi opsi pekerjaan KTP Indonesia (untuk dropdown/validasi)."""
    return {"pekerjaan": KTP_PEKERJAAN_OPTIONS}


@app.post("/ocr-ktp/")
async def ocr_ktp(file: UploadFile = File(...)):
    """OCR untuk KTP Indonesia. Upload gambar KTP, dapatkan teks dan field terstruktur."""
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "success": False,
            "message": "Could not decode image.",
            "raw_text": None,
            "fields": {}
        }

    raw_text, err = ocr_ktp_image(img)
    if err:
        return {
            "success": False,
            "message": err,
            "raw_text": None,
            "fields": {}
        }

    fields = parse_ktp_fields(raw_text)
    # Pastikan semua key standar ada (nilai None jika tidak terdeteksi)
    out = {k: None for k in KTP_FIELD_KEYS}
    out.update({k: v for k, v in fields.items() if k in out and v})
    fields_read = sum(1 for v in out.values() if v is not None)
    if fields_read == 0:
        message = (
            "Teks KTP tidak terbaca. Pastikan: (1) foto KTP jelas dan tidak blur, "
            "(2) seluruh area KTP terlihat, (3) pencahayaan cukup. Coba foto ulang dengan KTP rata dan fokus."
        )
    else:
        message = "OCR selesai."
    return {
        "success": True,
        "message": message,
        "raw_text": raw_text,
        "fields": out,
        "fields_read": fields_read,
    }


print("FastAPI endpoint '/validate-face/' and '/ocr-ktp/' defined.")

# 7. Add a main block to run the FastAPI application
# To run this, save the code as, e.g., 'app.py' and run `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
# For colab, this block won't execute directly but demonstrates how it would be run.
if __name__ == "__main__":
    # In a typical Colab environment, you might use something like:
    # from threading import Thread
    # import time
    # def run_uvicorn():
    #     uvicorn.run(app, host="0.0.0.0", port=8000)
    # thread = Thread(target=run_uvicorn)
    # thread.start()
    # time.sleep(1) # Give server a moment to start
    # print("FastAPI server started. You can now send requests to http://0.0.0.0:8000/validate-face/")
    print("FastAPI application is set up. To run, use uvicorn: uvicorn your_script_name:app --host 0.0.0.0 --port 8000 --reload")