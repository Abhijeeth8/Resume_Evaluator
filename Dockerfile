FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run res_helper.py --server.port=8501 --server.address=0.0.0.0