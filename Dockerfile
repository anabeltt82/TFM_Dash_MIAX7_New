FROM python:3.8

# set the working directory in the container
WORKDIR /code
# copy the dependencies file to the working directory
COPY requirements.txt .
# install dependencies
RUN pip install -r requirements.txt
#copy the content of the local src directory to the working directory
COPY src/ .
COPY .aws/ ./.aws
# Expose the API Port
ENV aws_access_key_id=AKIAXMTJSWINJS2Q64PZ
ENV aws_secret_access_key=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

EXPOSE 8080
# Run the server
CMD ["python", "run_app.py"]