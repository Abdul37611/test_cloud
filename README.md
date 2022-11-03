# Installation Instructions

Supports python 3.8+ on Linux/MacOS systems  

Upgrade python to 3.8

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
apt-get update
sudo apt-get install python3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

```

Make python3.8 default

```
sudo update-alternatives --config python3
```
Choose the option number that has path ```usr/bin/python3.8``` and Status as ```manual mode```

Install python3.8 venv

```
sudo apt install python3.8-venv
```

Create and activate the virtual environment

```
python3.8 -m venv venv
source venv/bin/activate
```

Navigate to the project directory
```
pip3 install -r requirements.txt

```

# Running the web app

```
streamlit run app.py
```
