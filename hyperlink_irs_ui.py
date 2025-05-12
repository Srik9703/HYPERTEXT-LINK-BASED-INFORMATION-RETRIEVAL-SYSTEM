import streamlit as st
import pandas as pd
from hyperlink_irs_core import HyperlinkIRS

# Set page config
st.set_page_config(
    page_title="Hyperlink IR System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the system with session state
if 'hir_system' not in st.session_state:
    st.session_state.hir_system = HyperlinkIRS()
    st.session_state.crawling_done = False
    st.session_state.selected_url = None

# Sidebar controls
with st.sidebar:
    st.header("🔗 Configuration")
    input_urls = st.text_area(
        "Enter seed URLs to crawl (one per line)",
        "https://en.wikipedia.org/wiki/PageRank\nhttps://en.wikipedia.org/wiki/HITS_algorithm",
        key='input_urls'
    )
    
    with st.expander("Crawling Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.slider("Max pages", 5, 50, 10, key='max_pages')
        with col2:
            max_depth = st.slider("Max depth", 1, 5, 2, key='max_depth')
        
        same_domain = st.checkbox("Stay on same domain", True, key='same_domain')
    
    algorithm = st.radio(
        "Ranking Algorithm",
        ("PageRank", "HITS"),
        index=0,
        key='algorithm'
    )
    
    damping = st.slider(
        "PageRank Damping",
        min_value=0.1,
        max_value=0.99,
        value=0.85,
        step=0.01,
        key='damping'
    )
    
    if st.button("Start Crawling & Ranking", type="primary", key='crawl_button'):
        urls = [url.strip() for url in st.session_state.input_urls.split('\n') if url.strip()]
        valid_urls = [url for url in urls if st.session_state.hir_system.is_valid_url(url)]
        
        if not valid_urls:
            st.error("Please enter at least one valid URL (with http:// or https://)")
        else:
            with st.spinner(f"Crawling {len(valid_urls)} starting URLs..."):
                processed = st.session_state.hir_system.crawl(
                    valid_urls,
                    max_pages=st.session_state.max_pages,
                    max_depth=st.session_state.max_depth,
                    same_domain=st.session_state.same_domain
                )
                
                if processed > 0:
                    st.success(f"Successfully crawled {processed} pages")
                    with st.spinner("Computing PageRank..."):
                        st.session_state.hir_system.compute_pagerank(damping=st.session_state.damping)
                    with st.spinner("Computing HITS scores..."):
                        st.session_state.hir_system.compute_hits()
                    st.session_state.crawling_done = True
                    st.session_state.selected_url = st.session_state.hir_system.get_all_crawled_urls()[0]
                    st.success("Ranking complete!")
                else:
                    st.warning("No pages were processed")

# Main content
st.title("🔗 Hyperlink-Based Information Retrieval System")

if st.session_state.crawling_done:
    # Move URL selection outside of tabs
    crawled_urls = [url for url in st.session_state.hir_system.page_index.keys()]
    if crawled_urls:
        selected_url = st.selectbox(
            "Select a URL to analyze",
            options=crawled_urls,
            index=0,
            key='url_select'
        )
        st.session_state.selected_url = selected_url
    
    tab1, tab2 = st.tabs(["URL Analysis", "Rankings"])

    with tab1:
        st.header("📄 URL Analysis")
        
        if not crawled_urls:
            st.warning("No valid URLs available for analysis")
        else:
            if st.session_state.selected_url:
                url_info = st.session_state.hir_system.search_by_url(st.session_state.selected_url)
                
                if not url_info:
                    st.error("No information available for this URL")
                else:
                    with st.container():
                        st.subheader(url_info['title'])
                        st.markdown(f"**URL:** {url_info['url']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PageRank", f"{url_info['pagerank']:.4f}")
                        with col2:
                            st.metric("Hub Score", f"{url_info['hub_score']:.4f}")
                        with col3:
                            st.metric("Authority Score", f"{url_info['auth_score']:.4f}")
                    
                    with st.expander("📝 Page Content", expanded=True):
                        st.write(url_info['content'])
                    
                    with st.expander(f"🔗 Outlinks ({len(url_info['outlinks'])})", expanded=True):
                        if url_info['outlinks']:
                            st.write("Links this page points to:")
                            for outlink, title, crawled in url_info['outlinks']:
                                if crawled:
                                    st.markdown(f"- 🟢 **[{title}]({outlink})** (Crawled)")
                                else:
                                    st.markdown(f"- 🔴 {outlink} (Not crawled)")
                        else:
                            st.write("No outlinks found")
                    
                    with st.expander(f"🔙 Backlinks ({len(url_info['backlinks'])})"):
                        if url_info['backlinks']:
                            st.write("Pages that link to this page:")
                            for backlink, title in url_info['backlinks']:
                                st.markdown(f"- [{title}]({backlink})")
                        else:
                            st.write("No backlinks found")
                    
                    with st.expander(f"📊 Similar Pages ({len(url_info['similar_pages'])})"):
                        if url_info['similar_pages']:
                            for similar_url, content in url_info['similar_pages']:
                                st.markdown(f"- [{similar_url}]({similar_url})")
                                st.caption(content)
                        else:
                            st.write("No similar pages found")

    with tab2:
        st.header("🏆 Page Rankings")
        
        # Show current selection context
        if st.session_state.selected_url:
            st.write(f"Currently selected: {st.session_state.selected_url}")
        
        if st.session_state.algorithm == "PageRank":
            if st.session_state.hir_system.page_ranks:
                pr_df = pd.DataFrame(
                    [(url, score) for url, score in st.session_state.hir_system.page_ranks.items()],
                    columns=["URL", "PageRank Score"]
                ).sort_values("PageRank Score", ascending=False)
                
                st.dataframe(
                    pr_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL"),
                        "PageRank Score": st.column_config.ProgressColumn(
                            "PageRank Score",
                            format="%.4f",
                            min_value=0,
                            max_value=pr_df["PageRank Score"].max()
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("PageRank not computed yet")
        else:
            if st.session_state.hir_system.hub_scores and st.session_state.hir_system.auth_scores:
                hits_df = pd.DataFrame(
                    [(url, st.session_state.hir_system.hub_scores[url], 
                      st.session_state.hir_system.auth_scores[url]) 
                     for url in st.session_state.hir_system.hub_scores.keys()],
                    columns=["URL", "Hub Score", "Authority Score"]
                ).sort_values("Authority Score", ascending=False)
                
                st.dataframe(
                    hits_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL"),
                        "Hub Score": st.column_config.ProgressColumn(
                            "Hub Score",
                            format="%.4f",
                            min_value=0,
                            max_value=hits_df["Hub Score"].max()
                        ),
                        "Authority Score": st.column_config.ProgressColumn(
                            "Authority Score",
                            format="%.4f",
                            min_value=0,
                            max_value=hits_df["Authority Score"].max()
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("HITS scores not computed yet")
else:
    st.info("ℹ️ Please enter seed URLs and click 'Start Crawling & Ranking' to begin")

with st.expander("ℹ️ About this System"):
    st.markdown("""
    ### Hyperlink-Based Information Retrieval System
    
    This system demonstrates:
    
    - **Web Crawling**: Follows links from seed URLs
    - **Link Analysis**:
      - PageRank: Measures page importance
      - HITS: Computes hub/authority scores
    - **Content Analysis**: TF-IDF for text relevance
    
    Features:
    - Complete URL information tracking
    - Comprehensive page rankings
    - Detailed URL analysis
    """)