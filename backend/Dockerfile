# FROM python:3.7-slim
# ADD . /code
# WORKDIR /code

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# EXPOSE 8000
# CMD ["python", "app.py"]

FROM python:3.7-slim
ADD . /code
WORKDIR /code


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]

