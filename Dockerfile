FROM python:3.11

RUN apt-get update && apt-get -y install graphviz

RUN pip install --no-cache-dir \
    scikit-learn==1.2.1 \
    scipy==1.11.2 \
    simpy==4.0.1 \
    pm4py==2.7.5.2 \
    statsmodels==0.14.0 \
    pandas==1.5.3

RUN git clone --branch 0.6.0 https://github.com/anyoptimization/pymoo /pymoo
WORKDIR /pymoo
RUN make compile
RUN pip install .

RUN echo 'alias ls="ls --color=auto"' > /root/.bashrc