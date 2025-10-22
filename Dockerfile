FROM python:3.9

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requierements_fast_api.txt

COPY ./start_fast_api.py /
COPY ./chatbot /

CMD ["fastapi", "run", "start_fast_api.py", "--port", "8000"]