# HYPERTEXT-LINK-BASED-INFORMATION-RETRIEVAL-SYSTEM

This project implements a hyperlink-based information retrieval system using PageRank and HITS (Hub & Authority) algorithms. It crawls web pages, analyzes their link structure, and ranks them based on their importance in the web graph. The system features a Streamlit-based web interface that allows interactive crawling, ranking, and URL analysis.

Features
--------
- Crawl web pages starting from one or more seed URLs.
- Choose between:
  - PageRank: Global link-based importance
  - HITS: Hub and authority scores
- View:
  - Page content
  - Outlinks (forward links)
  - Backlinks (incoming links)
  - Similar pages 
- Interactive UI with score metrics
- Option to restrict crawling to the same domain
- Dynamic selection of max depth and pages to crawl

Technologies Used
-----------------
- Python
- Streamlit

Additional required libraries:
- requests
- beautifulsoup4
- networkx
- scikit-learn
- pandas
- numpy

Project Structure
-----------------
├── hyperlink_irs_core.py     # Core logic: crawling, ranking, graph creation
├── hyperlink_irs_ui.py       # Streamlit app interface
├── requirements.txt          # Dependency list


How to Run
----------

1.Clone the repository

2. Create and activate a virtual environment (optional but recommended)
   python -m venv myenv
   myenv\Scripts\activate  (Windows) or source myenv/bin/activate (Linux/Mac)

3. Install dependencies
   pip install -r requirements.txt

4. Run the  application
   streamlit run hyperlink_irs_ui.py

5. Access the app
   Open your browser and go to http://127.0.0.1:8501/

Evaluation
----------
- Assess how well the PageRank and HITS algorithms surface authoritative or relevant pages in response to specific seed URLs.
- Compare outputs from PageRank vs HITS
- Manually inspect the top-ranked pages to verify their importance or quality.

Authors
-------
G.Pallavi(21071A05E5)
G.Vemkata Sai(21071A05E9)
K.Srija(21071A05F4) 
S.Vamshi Krishna(21071A05J2)

