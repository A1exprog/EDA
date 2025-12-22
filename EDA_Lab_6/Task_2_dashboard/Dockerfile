# FROM python3.11-slim

# WORKDIR app

# RUN pip install --no-cache-dir 
#     explainerdashboard 
#     scikit-learn 
#     pandas 
#     numpy 
#     matplotlib 
#     seaborn

# COPY . .

# RUN python generate_dashboard.py

# EXPOSE 9050

# CMD [python, run_dashboard.py]
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    explainerdashboard==0.4.3 \
    plotly==5.17.0 \
    dash==2.14.2 \
    "flask<3" \
    "jinja2<3.1" \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    waitress

COPY . .

RUN python dashboard/generate_dashboard.py

EXPOSE 9050

CMD ["python", "dashboard/run_dashboard.py"]


