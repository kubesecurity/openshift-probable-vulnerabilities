FROM python:3.6.3-slim
RUN pip3 install jira && pip3 install boto3
COPY ./ /src
RUN chgrp -R 0 /src \
  && chmod -R g+rwX /src
WORKDIR /src
CMD ["python", "./jira_data_collection.py"]