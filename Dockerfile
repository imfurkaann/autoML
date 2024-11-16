# Base image olarak Python kullan
FROM python:3.10

# Çalışma dizinini ayarla
WORKDIR /app

ENV FLASK_APP=web/run.py
ENV FLASK_RUN_HOST=0.0.0.0

# Gereken dosyaları kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Uygulama dosyasını kopyala
COPY . .

# Uygulamayı başlat
CMD ["flask", "run"]