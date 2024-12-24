# Tellco

## Overview
This project utilizes various Python dependencies to accomplish its goals. Follow the instructions below to set up your environment and get started.

---

## Prerequisites
Before proceeding, ensure you have the following installed:
- **Python**: Version 3.9 or later.
- **Pip**: Python's package manager.
- **Git**: (optional) if you're cloning this repository.

---

## Setup Instructions

### 1. Clone the Repository 
If the project is hosted on a Git repository, clone it using the following commands:
```bash
git clone https://github.com/yohannesalex/tellco
cd <project-folder>
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to isolate the dependencies for this project.
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Use the `requirements.txt` file to install all required Python packages.
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
After installing, ensure all dependencies are installed correctly:
```bash
pip list
```

---

## Running the Project

1. **Activate the virtual environment** (if not already activated):
   ```bash
   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```


## Requirements
The following dependencies are required and are listed in the `requirements.txt` file:

```
altair==5.2.0
annotated-types==0.6.0
appdirs==1.4.4
asttokens==2.4.1
attrs==23.2.0
beautifulsoup4==4.12.3
blinker==1.7.0
cachetools==5.3.3
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
constants==0.6.0
contourpy==1.2.0
cycler==0.12.1
dacite==1.8.1
decorator==5.1.1
et-xmlfile==1.1.0
exceptiongroup==1.2.0
executing==2.0.1
fonttools==4.49.0
frozendict==2.4.0
gitdb==4.0.11
GitPython==3.1.42
html5lib==1.1
htmlmin==0.1.12
idna==3.6
ImageHash==4.3.1
ipython==8.22.2
jedi==0.19.1
Jinja2==3.1.3
joblib==1.3.2
jsonschema==4.21.1
jsonschema-specifications==2023.12.1
kaleido==0.2.1
kiwisolver==1.4.5
llvmlite==0.41.1
lxml==5.1.0
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.8.3
matplotlib-inline==0.1.6
mdurl==0.1.2
multimethod==1.11.2
multitasking==0.0.11
networkx==3.2.1
numba==0.58.1
numpy==1.25.2
openpyxl==3.1.2
packaging==23.2
pandas==2.2.1
pandas-profiling==3.6.6
parso==0.8.3
patsy==0.5.6
peewee==3.17.1
phik==0.12.4
pillow==10.2.0
plotly==5.19.0
prompt-toolkit==3.0.43
protobuf==4.25.3
pure-eval==0.2.2
pyarrow==15.0.1
pydantic==2.6.3
pydantic_core==2.16.3
pydeck==0.8.1b0
Pygments==2.17.2
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyWavelets==1.5.0
PyYAML==6.0.1
pyzmq==25.1.2
referencing==0.33.0
requests==2.31.0
rich==13.7.1
rpds-py==0.18.0
scikit-learn==1.4.1.post1
scipy==1.11.4
seaborn==0.12.2
six==1.16.0
smmap==5.0.1
soupsieve==2.5
stack-data==0.6.3
statsmodels==0.14.1
streamlit==1.32.0
streamlit-pandas-profiling==0.1.3
tangled-up-in-unicode==0.2.0
tenacity==8.2.3
threadpoolctl==3.3.0
toml==0.10.2
toolz==0.12.1
tornado==6.4
tqdm==4.66.2
traitlets==5.14.2
typeguard==4.1.5
typing_extensions==4.10.0
tzdata==2024.1
urllib3==2.2.1
visions==0.7.5
watchdog==4.0.0
wcwidth==0.2.13
webencodings==0.5.1
wordcloud==1.9.3
ydata-profiling==4.6.5
yfinance==0.2.37
```

