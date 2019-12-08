#!/bin/bash
GREEN='\033[0;32m'
NC='\033[0m'
echo -e "${GREEN}[Info] Start installation${NC}"
pip install -r requirements.txt
echo -e "${GREEN}[Info] Install complete, now download nltk dataset${NC}"
python -c "import nltk; nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('punkt'); nltk.download('reuters')"
echo -e "${GREEN}[Info] Installing corenlp${NC}"
giveme5w1h-corenlp install
echo -e "${GREEN}[Info] All done.${NC}"
