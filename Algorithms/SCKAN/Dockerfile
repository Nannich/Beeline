FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

USER root

RUN apt-get update && apt-get install -y git time && rm -rf /var/lib/apt/lists/*

RUN pip install pandas==2.2.1 shap==0.45.1 numpy==1.23.5

RUN pip install git+https://github.com/Blealtan/efficient-kan.git

RUN git clone https://github.com/1250326/scKAN.git /app
WORKDIR /app

ENTRYPOINT ["python", "KAN.py"]