FROM python:3.13.0

COPY index.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 80

ENV NAME CaseScaling

CMD ["python", "index.py"]