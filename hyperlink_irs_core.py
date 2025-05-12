import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import sqlite3
from collections import defaultdict
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

class ThreadSafeDB:
    """Thread-safe SQLite database wrapper"""
    def __init__(self):
        self.local = threading.local()
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize database connection for current thread"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(':memory:', check_same_thread=False)
            self.local.cursor = self.local.conn.cursor()
            self.local.cursor.execute('''CREATE TABLE IF NOT EXISTS pages
                                 (id INTEGER PRIMARY KEY, url TEXT UNIQUE, title TEXT, content TEXT)''')
            self.local.cursor.execute('''CREATE TABLE IF NOT EXISTS links
                                 (source INTEGER, target INTEGER, 
                                 FOREIGN KEY(source) REFERENCES pages(id),
                                 FOREIGN KEY(target) REFERENCES pages(id))''')
            self.local.conn.commit()
    
    def execute(self, query, params=None):
        """Execute SQL query"""
        self.initialize_db()
        try:
            if params:
                return self.local.cursor.execute(query, params)
            return self.local.cursor.execute(query)
        except sqlite3.IntegrityError:
            return None
    
    def commit(self):
        """Commit changes"""
        self.local.conn.commit()
    
    def fetchall(self):
        """Fetch all results"""
        return self.local.cursor.fetchall()
    
    def fetchone(self):
        """Fetch one result"""
        return self.local.cursor.fetchone()
    
    @property
    def lastrowid(self):
        """Get last inserted row ID"""
        return self.local.cursor.lastrowid

class HyperlinkIRS:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.page_ranks = None
        self.hub_scores = None
        self.auth_scores = None
        self.page_index = {}
        self.reverse_index = {}
        self.content_index = {}
        self.db = ThreadSafeDB()
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.lock = threading.Lock()
        
    def is_valid_url(self, url):
        """Check if URL is valid and has http/https scheme"""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc]) and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def get_domain(self, url):
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc
    
    def fetch_page(self, url, max_retries=3):
        """Fetch and parse a web page"""
        for _ in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    return None
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img']):
                    element.decompose()
                
                title = soup.title.string if soup.title else url
                text = ' '.join(soup.stripped_strings)
                
                return {
                    'url': url,
                    'title': title,
                    'content': text,
                    'html': str(soup)
                }
            except Exception as e:
                time.sleep(2)
        return None
    
    def extract_links(self, url, html_content):
        """Extract all links from a page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(url, href)
            
            if self.is_valid_url(absolute_url) and not href.startswith('#'):
                links.add(absolute_url)
                
        return links
    
    def add_page(self, url, title, content):
        """Add a page to the database"""
        with self.lock:
            self.db.execute("SELECT id FROM pages WHERE url=?", (url,))
            existing = self.db.fetchone()
            
            if existing:
                return existing[0]
                
            self.db.execute("INSERT INTO pages (url, title, content) VALUES (?, ?, ?)", 
                          (url, title, content))
            self.db.commit()
            page_id = self.db.lastrowid
            self.page_index[url] = page_id
            self.reverse_index[page_id] = url
            self.content_index[page_id] = content.lower()
            self.graph.add_node(url, title=title, content=content)
            return page_id
    
    def add_link(self, source_url, target_url):
        """Add a link between pages"""
        with self.lock:
            self.db.execute("SELECT id FROM pages WHERE url=?", (source_url,))
            source_id = self.db.fetchone()
            self.db.execute("SELECT id FROM pages WHERE url=?", (target_url,))
            target_id = self.db.fetchone()
            
            if source_id and target_id:
                source_id = source_id[0]
                target_id = target_id[0]
                
                self.db.execute("SELECT 1 FROM links WHERE source=? AND target=?", 
                              (source_id, target_id))
                if self.db.fetchone():
                    return True
                
                self.db.execute("INSERT INTO links (source, target) VALUES (?, ?)", 
                              (source_id, target_id))
                self.db.commit()
                self.graph.add_edge(source_url, target_url)
                return True
            return False
    
    def crawl(self, start_urls, max_pages=10, max_depth=2, same_domain=True):
        """Crawl websites starting from given URLs"""
        queue = [(url, 0) for url in start_urls if self.is_valid_url(url)]
        processed = 0
        
        while queue and processed < max_pages:
            url, depth = queue.pop(0)
            
            if url in self.visited_urls or depth > max_depth:
                continue
                
            self.visited_urls.add(url)
            
            page_data = self.fetch_page(url)
            if not page_data:
                continue
                
            page_id = self.add_page(url, page_data['title'], page_data['content'])
            processed += 1
            
            links = self.extract_links(url, page_data['html'])
            
            if same_domain:
                domain = self.get_domain(url)
                links = [link for link in links if self.get_domain(link) == domain]
            
            for link in links:
                if link not in self.visited_urls and link not in [u for u, _ in queue]:
                    queue.append((link, depth + 1))
                
                if link in self.page_index or link in [u for u, _ in queue]:
                    self.add_link(url, link)
        
        if processed > 0:
            self.build_tfidf_matrix()
        
        return processed
    
    def build_tfidf_matrix(self):
        """Build TF-IDF matrix for all documents"""
        with self.lock:
            self.db.execute("SELECT content FROM pages")
            documents = [row[0] for row in self.db.fetchall()]
        
        if documents:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
    def compute_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        """Compute PageRank scores"""
        if len(self.graph) == 0:
            return None
            
        adj_matrix = nx.adjacency_matrix(self.graph)
        n = adj_matrix.shape[0]
        
        out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1
        
        transition = adj_matrix.multiply(1 / out_degree).transpose()
        
        ranks = np.ones(n) / n
        dangling = (out_degree == 1)
        
        for _ in range(max_iter):
            prev_ranks = ranks.copy()
            ranks = (damping * (transition @ ranks) + 
                     (1 - damping) / n + 
                     damping * np.sum(ranks[dangling]) / n)
            
            if np.linalg.norm(ranks - prev_ranks, 1) < tol:
                break
        
        self.page_ranks = {url: ranks[i] for i, url in enumerate(self.graph.nodes())}
        return self.page_ranks
    
    def compute_hits(self, max_iter=50, tol=1e-6):
        """Compute HITS hub and authority scores"""
        if len(self.graph) == 0:
            return None, None
            
        adj_matrix = nx.adjacency_matrix(self.graph)
        n = adj_matrix.shape[0]
        
        hubs = np.ones(n)
        authorities = np.ones(n)
        
        for _ in range(max_iter):
            prev_hubs = hubs.copy()
            prev_auth = authorities.copy()
            
            authorities = adj_matrix.transpose() @ hubs
            auth_norm = np.linalg.norm(authorities)
            if auth_norm > 0:
                authorities /= auth_norm
                
            hubs = adj_matrix @ authorities
            hub_norm = np.linalg.norm(hubs)
            if hub_norm > 0:
                hubs /= hub_norm
                
            if (np.linalg.norm(hubs - prev_hubs, 1) < tol and
                np.linalg.norm(authorities - prev_auth, 1) < tol):
                break
        
        self.hub_scores = {url: hubs[i] for i, url in enumerate(self.graph.nodes())}
        self.auth_scores = {url: authorities[i] for i, url in enumerate(self.graph.nodes())}
        return self.hub_scores, self.auth_scores
    
    def search_by_url(self, url):
        """Get comprehensive information about a specific URL"""
        # Normalize the URL first
        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized_url.endswith('/'):
            normalized_url = normalized_url[:-1]
        
        with self.lock:
            # Check both exact and normalized versions
            self.db.execute("SELECT id FROM pages WHERE url=? OR url=?", (url, normalized_url))
            page_id = self.db.fetchone()
            
            if not page_id:
                return None
                
            page_id = page_id[0]
            
            # Get page data using the ID we found
            self.db.execute("SELECT title, content FROM pages WHERE id=?", (page_id,))
            page_data = self.db.fetchone()
            
            if not page_data:
                return None
                
            title, content = page_data
            
            # Get outlinks using the page ID
            self.db.execute('''SELECT p.url, p.title, 
                            CASE WHEN p.url IS NOT NULL THEN 1 ELSE 0 END as crawled
                            FROM links l
                            LEFT JOIN pages p ON l.target = p.id
                            WHERE l.source = ?''', (page_id,))
            outlinks = self.db.fetchall() or []
            
            # Get backlinks using the page ID
            self.db.execute('''SELECT p.url, p.title 
                            FROM pages p JOIN links l ON p.id = l.source 
                            WHERE l.target = ?''', (page_id,))
            backlinks = self.db.fetchall() or []
        
        # Get scores with proper defaults
        pagerank = self.page_ranks.get(url, self.page_ranks.get(normalized_url, 0)) if hasattr(self, 'page_ranks') else 0
        hub = self.hub_scores.get(url, self.hub_scores.get(normalized_url, 0)) if hasattr(self, 'hub_scores') else 0
        auth = self.auth_scores.get(url, self.auth_scores.get(normalized_url, 0)) if hasattr(self, 'auth_scores') else 0
        
        # Get similar pages
        similar_pages = []
        if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            try:
                idx = list(self.reverse_index.keys()).index(page_id)
                if idx < self.tfidf_matrix.shape[0]:
                    cosine_sim = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix)
                    sim_scores = [(i, score) for i, score in enumerate(cosine_sim[0]) if i != idx]
                    sim_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, score in sim_scores[:3]:  # Top 3 similar pages
                        if i < len(self.reverse_index):
                            similar_url = self.reverse_index[list(self.reverse_index.keys())[i]]
                            self.db.execute("SELECT content FROM pages WHERE id=?", (list(self.reverse_index.keys())[i],))
                            content = self.db.fetchone()
                            if content:
                                similar_pages.append((similar_url, content[0][:100] + "..."))
            except Exception as e:
                print(f"Error finding similar pages: {e}")
        
        return {
            'url': normalized_url,
            'title': title,
            'content': content,
            'backlinks': backlinks,
            'outlinks': outlinks,
            'pagerank': pagerank,
            'hub_score': hub,
            'auth_score': auth,
            'similar_pages': similar_pages
        }
    def get_all_crawled_urls(self):
        """Get all URLs that have been successfully crawled"""
        with self.lock:
            self.db.execute("SELECT url FROM pages")
            return [row[0] for row in self.db.fetchall()] or []
    
    def get_graph_data(self):
        """Get data for graph visualization"""
        if len(self.graph) == 0:
            return None
            
        nodes = []
        for url in self.graph.nodes():
            nodes.append({
                'id': url,
                'label': url.split('/')[-1],
                'title': self.graph.nodes[url].get('title', ''),
                'pagerank': self.page_ranks.get(url, 0) if hasattr(self, 'page_ranks') and self.page_ranks else 0,
                'hub': self.hub_scores.get(url, 0) if hasattr(self, 'hub_scores') and self.hub_scores else 0,
                'auth': self.auth_scores.get(url, 0) if hasattr(self, 'auth_scores') and self.auth_scores else 0
            })
        
        links = [{'source': u, 'target': v} for u, v in self.graph.edges()]
        
        return {'nodes': nodes, 'links': links}