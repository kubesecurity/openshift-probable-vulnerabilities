FROM registry.centos.org/centos/centos:7

RUN yum install -y epel-release &&\
    yum install -y gcc git python36-pip python36-requests httpd httpd-devel python36-devel postgresql postgresql-devel &&\
    yum clean all

COPY src/requirements.txt /

RUN pip3 install --upgrade pip>=10.0.0 &&\
    pip3 install -r requirements.txt && rm requirements.txt

COPY src/  /src
COPY swagger/ /swagger

WORKDIR /src

CMD ["python3", "rest_api.py"]