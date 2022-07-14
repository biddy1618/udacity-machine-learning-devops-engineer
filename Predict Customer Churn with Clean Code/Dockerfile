FROM python:3.8-slim
COPY ./ /churn_project/
WORKDIR /churn_project/
# RUN apt-get update && apt-get install -y libpq-dev gcc

RUN pip install --upgrade pip==22.0.3
RUN pip install -r requirements_py3.8_local.txt
RUN ["bash", "-c" "pytest --disable-warnings churn_script_logging_and_tests.py"]
RUN ["bash", "-c", "ipython churn_library.py"]
ENTRYPOINT ["bash"]
