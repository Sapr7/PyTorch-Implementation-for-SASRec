# 🧠 SASRec: PyTorch Implementation for Sequential Recommendation

Реализация модели [SASRec](https://arxiv.org/abs/1808.09781) — Self-Attentive Sequential Recommendation — на PyTorch с использованием PyTorch Lightning, Hydra, DVC, ONNX и MLflow.

---

## ⚙️ Setup

1. **Клонируйте репозиторий:**

```bash
git clone git@github.com:Sapr7/PyTorch-Implementation-for-SASRec.git
cd pytorch-sasrec
```

2. **Создайте виртуальное окружение и установите зависимости через [uv](https://github.com/astral-sh/uv):**

```bash
uv venv
uv pip install
```

> ⚠️ Требуется Python ≥ 3.13 и установленный `uv`
> Установить: `pip install uv`

3. **Загрузите данные с помощью DVC:**

```bash
dvc pull
```

---

## 🏋️‍♀️ Train

Запуск обучения:

```bash
python -m scripts.train
```

📁 Используется конфигурация `configs/train.yaml`, которая ссылается на:
- общие параметры в `configs/config.yaml`
- модель SASRec
- данные: `data/ml-1m.txt`
- директория чекпойнтов: `checkpoints/ml-1m_default/`

🛠 Примеры параметров:

```yaml
batch_size: 128
lr: 0.001
maxlen: 50
num_epochs: 20
```

---

## 🔎 MLflow UI

Все метрики и параметры обучения логируются в директорию `mlruns/`.

Для запуска веб-интерфейса MLflow:

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Затем откройте в браузере:

```
http://localhost:5000
```

> 💡 Можно изменить порт или использовать `--host 0.0.0.0` для удалённого доступа.

---

## 🧪 Inference

### 💾 Формат входных данных

Файл `test_input.txt`, где каждая строка — пара `user_id item_id`:

```
1 10
1 20
1 35
2 50
2 60
```

---

### 🚀 Запуск инференса

```bash
python -m scripts.inference
```

📁 Конфигурация `configs/infer.yaml` указывает:

```yaml
input_txt: data/test_input.txt
output_path: output/recommendations.json
save_dir: checkpoints/ml-1m_default
```

📝 Пример результата (`recommendations.json`):

```json
{
  "1": [42, 13, 17, 55, 88],
  "2": [61, 44, 32, 79, 11]
}
```

---

## 🧩 Production Preparation

### Экспорт в ONNX / TensorRT:

```bash
python -m scripts.convert_and_export
```

📦 Что создаётся:
- `checkpoints/ml-1m_default/sasrec.onnx`
- `checkpoints/ml-1m_default/sasrec.trt` (если доступен TensorRT)
- `checkpoints/ml-1m_default/last.ckpt`

Эти файлы можно использовать в продакшн-инференсе без PyTorch.

---

## 📁 Состав проекта

```
├── scripts/          # train.py, inference.py, convert_and_export.py
├── model/            # SASRec, Lightning-модуль, attention-блоки
├── utils/            # загрузка данных, валидация, экспорт
├── configs/          # Hydra-конфиги
├── data/             # данные под DVC
├── checkpoints/      # чекпойнты и onnx/tensorrt модели
├── output/           # json с результатами предсказания
├── mlruns/           # метрики обучения (MLflow)
├── pyproject.toml    # зависимости (uv)
├── uv-pyproject.lock # фиксированные версии
```

---

## 📎 Зависимости

- `python >= 3.13`
- `uv`
- `torch >= 2.7`
- `pytorch-lightning >= 2.5`
- `hydra-core`, `omegaconf`
- `mlflow`
- `dvc`, `dvc-gdrive`
- `onnx`, `onnxruntime`
- `tensorrt` (опционально)
- `pycuda` (опционально)

---

## 📜 License

MIT License.
