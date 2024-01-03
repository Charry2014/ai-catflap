FROM python:3.10

RUN pip install flask flask_socketio
COPY app.py .
RUN mkdir templates
COPY ./templates/* ./templates/

ENTRYPOINT ["python", "app.py"]