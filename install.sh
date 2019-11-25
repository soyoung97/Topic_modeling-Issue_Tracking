#!/bin/bash
echo "[Info] Start installation"
pip install -r requirements.txt
echo "[Info] Install complete, now downlad nltk dataset"
python -c "import nltk; nltk.download('stopwords')"
echo "[Info] All done."
