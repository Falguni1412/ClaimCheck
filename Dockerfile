FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
COPY . .
# bake the model into the image so startup never depends on the network
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; n='cross-encoder/nli-MiniLM2-L6-H768'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)"
RUN chgrp -R 0 /opt/hf /app && chmod -R g=u /opt/hf /app
EXPOSE 8080
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8080","--workers","1"]