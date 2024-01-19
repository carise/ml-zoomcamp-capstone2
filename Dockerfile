FROM public.ecr.aws/lambda/python:3.11

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp311-cp311-linux_x86_64.whl

COPY ["lambda_function.py", "model.tflite", "./"]

CMD [ "lambda_function.lambda_handler" ]
