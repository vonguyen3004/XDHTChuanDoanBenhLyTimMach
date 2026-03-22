---
title: Ecg Ai Flask
emoji: "🫀"
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# He thong AI chan doan dien tam do ECG

Ung dung Flask ho tro:
- phan tich ECG bang AI
- xem lich su chan doan
- bao cao va danh gia mo hinh
- chay local bang Docker ma khong can cai Python, TensorFlow hay moi truong ao

## Link server

Server dang chay tai:

`https://zn3004-ecg-ai-flask.hf.space`

## Chay nhanh bang Docker

Yeu cau duy nhat:
- da cai `Docker Desktop`

Sau khi tai source code ve, mo terminal tai thu muc du an va chay:

```bash
docker compose up --build
```

Sau khi container khoi dong xong, mo:

`http://localhost:7860`

Voi cach nay:
- Flask app chay trong container `web`
- MongoDB chay trong container `mongo`
- khong can cai them Python package hay tao virtual environment

## Dung lai he thong

```bash
docker compose down
```

Neu muon xoa ca du lieu Mongo local:

```bash
docker compose down -v
```

## Cau truc Docker

- `Dockerfile`: dong goi ung dung Flask
- `docker-compose.yml`: chay cung luc web + MongoDB
- `.dockerignore`: loai bo file cache/log/venv khi build

## Chay khong dung Docker

Neu can chay thu cong:

```bash
python backend/app.py
```

Nhung cach khuyen nghi de nop va demo la Docker Compose vi de tai ve va chay ngay.

## Ghi chu

- Khi chay local bang Docker Compose, ung dung tu dung MongoDB local trong container:
  `mongodb://mongo:27017`
- Khi deploy Hugging Face Space, app dung bien moi truong `MONGO_URI`
- Mo hinh AI va file report da duoc dong kem trong repo

## GitHub

Repo nay da duoc chuan bi theo huong:
- clone ve
- chay `docker compose up --build`
- truy cap `http://localhost:7860`

Khong can tao moi truong ao hay cai thu vien Python bang tay.
