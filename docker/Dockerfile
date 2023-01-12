FROM python:3.7-slim
WORKDIR /api
COPY . .
COPY ./docker/api_requirements.txt ./requirements.txt
RUN rm -r ./docker
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]
EXPOSE 5001