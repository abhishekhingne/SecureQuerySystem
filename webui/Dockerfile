#1. Pull Container Image
FROM ubuntu:22.04

#2. Update the packages
# RUN apt upgrade
RUN apt update --fix-missing
RUN apt -y upgrade
RUN apt-get install -y software-properties-common libgl1 wget python3.11 python3-pip poppler-utils


#4. Install pip for python3.11
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.11 get-pip.py
RUN python3.11 -m pip install --upgrade pip

#5. Copy AppCode to Container and change workdir application home
COPY ./src /opt/app/streamlit
WORKDIR /opt/app/streamlit

#6. Install App(Python) Dependencies from PyPi
RUN pip3 install --ignore-installed -r requirements.txt

#7. Run the API
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]