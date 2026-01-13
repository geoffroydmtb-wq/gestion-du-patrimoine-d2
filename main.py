import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import feedparser
import requests
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURATION GLOBALE ---
st.set_page_config(
    page_title="Wealth Manager Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN SYSTEM "PURE & GOLD" ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #1F1F1F; }
    
    /* HEADER & ARROWS */
    header[data-testid="stHeader"] { background-color: transparent !important; }
    button[kind="header"] { color: #D4AF37 !important; }
    [data-testid="stSidebarCollapsedControl"] { color: #D4AF37 !important; display: block !important; }

    /* TYPOGRAPHY */
    h1, h2, h3 { color: #D4AF37 !important; font-weight: 300; letter-spacing: -0.5px; }
    h4, h5 { color: #A0A0A0 !important; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; }

    /* CARDS */
    div[data-testid="metric-container"], .news-card, .macro-card {
        background-color: #161A25;
        border: 1px solid #252A38;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    div[data-testid="metric-container"]:hover, .news-card:hover, .macro-card:hover {
        border-color: #D4AF37;
    }
    
    /* NEWS STYLE */
    .news-title { font-size: 16px; font-weight: 600; color: #FAFAFA; margin-bottom: 5px; }
    .news-date { font-size: 12px; color: #D4AF37; margin-bottom: 10px; }
    .news-summary { font-size: 14px; color: #B0B0B0; }
    .news-link { color: #D4AF37; text-decoration: none; font-size: 14px; }

    /* INPUTS & BUTTONS */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #161A25; border: 1px solid #252A38; color: #FAFAFA; border-radius: 8px;
    }
    .stButton>button {
        width: 100%; background-color: transparent; border: 1px solid #D4AF37; color: #D4AF37; border-radius: 8px;
    }
    .stButton>button:hover { background-color: #D4AF37; color: #000000; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SÉCURITÉ ---
def check_password():
    if st.secrets.get("password") is None: return True
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False): return True
    st.markdown("<br><br><h1 style='text-align: center; color:#D4AF37;'>SECURE ACCESS</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2: st.text_input("Access Key", type="password", on_change=password_entered, key="password")
    return False

if not check_password(): st.stop()

# --- INITIALISATION ---
if 'journal_ordres' not in st.session_state:
    st.session_state.journal_ordres = pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Quantité', 'Prix Unitaire', 'Frais', 'Total'])

try: from constantes import CATALOGUE
except ImportError: CATALOGUE = {}

# --- FONCTIONS UTILITAIRES ---

def get_readable_name(ticker):
    """Traduit 'AAPL' en 'AAPL (Apple Inc.)' pour l'affichage"""
    if not CATALOGUE: return ticker
    for cat in CATALOGUE.values():
        for name, code in cat.items():
            if code == ticker:
                # Le nom dans le catalogue est souvent "Apple (AAPL)"
                # On veut extraire juste "Apple"
                clean_name = name.split(" (")[0]
                return f"{ticker} ({clean_name})"
    return ticker

def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#A0A0A0", family="Inter"),
        margin=dict(t=30, l=0, r=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#252A38', zeroline=False),
        colorway=["#D4AF37", "#FFFFFF", "#808080", "#555555"]
    )
    return fig

@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = { "VIX (Peur)": "^VIX", "Taux US 10Y": "^TNX", "Or ($/oz)": "GC=F", "Pétrole ($)": "CL=F", "EUR/USD": "EURUSD=X" }
    try:
        data = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
        macro_info = {}
        for name, ticker in tickers.items():
            try:
                if isinstance(data.columns, pd.MultiIndex): series = data[ticker]
                elif ticker in data.columns: series = data[ticker]
                else: series = data.iloc[:, 0]
                last = float(series.iloc[-1])
                prev = float(series.iloc[-2])
                delta = ((last - prev) / prev) * 100
                macro_info[name] = (last, delta)
            except: macro_info[name] = (0.0, 0.0)
        return macro_info
    except: return {}

@st.cache_data
def get_stock_data_optimized(tickers, start, end):
    if not tickers: return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
    except Exception: return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    
    target_col = 'Adj Close'
    if target_col not in df.columns:
        if 'Close' in df.columns: target_col = 'Close'
        else: return pd.DataFrame()

    data = df[target_col]
    if len(tickers) == 1:
        if isinstance(data, pd.Series): data = data.to_frame(); data.columns = tickers
        elif isinstance(data, pd.DataFrame) and data.shape[1] == 1: data.columns = tickers
    return data

def run_monte_carlo_simulation(df_prices, num_portfolios=2000):
    returns = df_prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(df_prices.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_ret = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = p_ret
        results[1,i] = p_std
        results[2,i] = p_ret / p_std
        
    return results, weights_record

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### WEALTH MANAGER")
    st.markdown("---")
    menu_selection = st.radio("NAVIGATION", ["Tableau de Bord", "Marchés & Analyse", "Transactions", "Actualités & Infos"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("v2.2 • Clear Names")

# ==============================================================================
# PAGE 1 : TABLEAU DE BORD
# ==============================================================================
if menu_selection == "Tableau de Bord":
    st.title("Synthèse Patrimoniale")
    with st.expander("📝 Mettre à jour mes soldes", expanded=True):
        c1, c2, c3 = st.columns(3)
        livret_a = c1.number_input("Liquidités (€)", value=10000.0, step=100.0)
        bourse = c2.number_input("Bourse (€)", value=5000.0, step=100.0)
        crypto = c3.number_input("Crypto (€)", value=1000.0, step=100.0)
    
    total_wealth = livret_a + bourse + crypto
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Valeur Nette", f"{total_wealth:,.0f} €")
    c2.metric("Liquidités", f"{livret_a:,.0f} €")
    c3.metric("Actifs Risqués", f"{bourse+crypto:,.0f} €", f"{((bourse+crypto)/total_wealth)*100:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns([1, 2])
    with c_chart1:
        st.markdown("#### RÉPARTITION")
        fig = px.pie(values=[livret_a, bourse, crypto], names=['Liquidités', 'Bourse', 'Crypto'], hole=0.7, color_discrete_sequence=['#333333', '#D4AF37', '#FAFAFA'])
        fig.update_traces(textinfo='none')
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(style_plotly(fig), use_container_width=True)
    with c_chart2:
        st.markdown("#### PROJECTION")
        c_a, c_b = st.columns(2)
        apport = c_a.number_input("Apport/mois", value=500.0)
        taux = c_b.slider("Rendement (%)", 0, 15, 6) / 100
        valeurs = [total_wealth]
        for _ in range(15): valeurs.append(valeurs[-1] * (1 + taux) + (apport * 12))
        df_proj = pd.DataFrame({'Année': range(16), 'Capital': valeurs})
        fig_area = px.area(df_proj, x='Année', y='Capital')
        fig_area.update_traces(line_color='#D4AF37', fillcolor='rgba(212, 175, 55, 0.1)')
        st.plotly_chart(style_plotly(fig_area), use_container_width=True)

# ==============================================================================
# PAGE 2 : MARCHÉS & ANALYSE
# ==============================================================================
elif menu_selection == "Marchés & Analyse":
    
    st.markdown("### INDICATEURS MACRO-ÉCONOMIQUES")
    macro_data = get_macro_data()
    if macro_data:
        cols_macro = st.columns(len(macro_data))
        for i, (name, (val, delta)) in enumerate(macro_data.items()):
            cols_macro[i].metric(name, f"{val:.2f}", f"{delta:+.2f}%")
    else: st.info("Chargement macro...")
    
    st.markdown("---")
    st.title("Analyse de Portefeuille")
    
    col_input, col_period = st.columns([3, 1])
    with col_input:
        st.markdown("#### SÉLECTION D'ACTIFS")
        selected_tickers = []
        if CATALOGUE:
            for cat, assets in CATALOGUE.items():
                sel = st.multiselect(f"{cat}", list(assets.keys()))
                for nom in sel: selected_tickers.append(assets[nom])
        manual = st.text_input("Ajout Manuel (Ticker)", placeholder="Ex: AI.PA")
        if manual: selected_tickers.append(manual.upper())
        tickers = list(set(selected_tickers))

    with col_period:
        st.markdown("#### PÉRIODE")
        start_date = st.date_input("Depuis", value=datetime.now() - timedelta(days=365*2))

    if tickers:
        with st.spinner('Analyse des données...'):
            df_prices = get_stock_data_optimized(tickers, start_date, datetime.now())
        
        if not df_prices.empty:
            
            tab_alloc, tab_optim = st.tabs(["📊 ALLOCATION MANUELLE", "🧠 OPTIMISATION MARKOWITZ"])
            
            with tab_alloc:
                st.markdown("#### ALLOCATION")
                cols = st.columns(4)
                weights = []
                found = df_prices.columns.tolist()
                for i, t in enumerate(found):
                    # NOUVEAU : On utilise get_readable_name pour l'affichage
                    label = get_readable_name(t)
                    with cols[i % 4]: weights.append(st.number_input(label, 0.0, 1.0, 1.0/len(found), 0.05, key=f"w_{t}"))
                
                use_benchmark = st.checkbox("Comparer au S&P 500 (SPY)", value=True)
                
                df_norm = (df_prices / df_prices.iloc[0]) * 100
                portf_ret = df_prices.pct_change().dropna().dot(weights)
                portf_cum = (1 + portf_ret).cumprod() * 100
                df_final = df_norm.copy()
                df_final['PORTFOLIO'] = portf_cum.fillna(100)
                
                if use_benchmark:
                    try:
                        bench = get_stock_data_optimized(['SPY'], start_date, datetime.now())
                        if not bench.empty:
                            bench_norm = (bench / bench.iloc[0]) * 100
                            df_final['S&P 500'] = bench_norm
                    except: pass
                
                colors = ["#D4AF37" if c == "PORTFOLIO" else "#FAFAFA" if c == "S&P 500" else "#333333" for c in df_final.columns]
                st.line_chart(df_final, color=colors)
                
                c_m1, c_m2 = st.columns(2)
                from logic.metrics import calculate_key_metrics, get_correlation_matrix
                with c_m1:
                    st.markdown("#### RISQUE")
                    st.dataframe(calculate_key_metrics(pd.DataFrame({'Portfolio': df_final['PORTFOLIO']})).T.style.format("{:.2f}"))
                with c_m2: 
                    st.markdown("#### CORRÉLATION")
                    # NOUVEAU : On renomme les colonnes et index pour l'affichage
                    corr = get_correlation_matrix(df_prices)
                    corr.index = [get_readable_name(t) for t in corr.index]
                    corr.columns = [get_readable_name(t) for t in corr.columns]
                    st.dataframe(corr.style.background_gradient(cmap='cividis', axis=None).format("{:.2f}"))
            
            with tab_optim:
                st.markdown("#### FRONTIÈRE EFFICIENTE (MONTE CARLO)")
                if st.button("Lancer l'Optimisation"):
                    results, weights_record = run_monte_carlo_simulation(df_prices)
                    max_sharpe_idx = np.argmax(results[2])
                    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
                    best_weights = weights_record[max_sharpe_idx]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results[1,:], y=results[0,:], mode='markers', marker=dict(color=results[2,:], colorscale='Cividis', size=5), name='Portefeuilles'))
                    fig.add_trace(go.Scatter(x=[sdp], y=[rp], mode='markers', marker=dict(color='#D4AF37', size=15, line=dict(width=2, color='white')), name='Max Sharpe'))
                    fig.update_layout(title='Frontière Efficiente', xaxis_title='Volatilité', yaxis_title='Rendement', font=dict(color='#A0A0A0'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### ALLOCATION OPTIMALE")
                    # NOUVEAU : Noms complets dans le tableau d'optimisation
                    full_names = [get_readable_name(t) for t in df_prices.columns]
                    opt_df = pd.DataFrame({'Actif': full_names, 'Poids Idéal': best_weights})
                    opt_df['Poids Idéal'] = opt_df['Poids Idéal'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(opt_df.T)

# ==============================================================================
# PAGE 3 : TRANSACTIONS
# ==============================================================================
elif menu_selection == "Transactions":
    st.title("Journal des Transactions")
    c1, c2 = st.columns(2)
    with c1: st.download_button("💾 Sauvegarder CSV", data=st.session_state.journal_ordres.to_csv(index=False).encode('utf-8'), file_name="journal.csv", mime="text/csv")
    with c2: 
        up = st.file_uploader("📂 Charger CSV", type=['csv'], label_visibility="collapsed")
        if up: st.session_state.journal_ordres = pd.read_csv(up); st.success("Chargé !")

    st.markdown("---")
    st.markdown("#### NOUVEL ORDRE")
    with st.form("new_order"):
        cols = st.columns(5)
        d = cols[0].date_input("Date")
        t = cols[1].text_input("Ticker", placeholder="AAPL")
        s = cols[2].selectbox("Sens", ["Achat", "Vente"])
        q = cols[3].number_input("Qté", min_value=0.01)
        p = cols[4].number_input("Prix Unit.", min_value=0.01)
        if st.form_submit_button("VALIDER"):
            new = {'Date': d, 'Ticker': t.upper(), 'Type': s, 'Quantité': q, 'Prix Unitaire': p, 'Frais': 0, 'Total': q*p}
            st.session_state.journal_ordres = pd.concat([st.session_state.journal_ordres, pd.DataFrame([new])], ignore_index=True)
            st.rerun()

    st.markdown("#### HISTORIQUE")
    st.dataframe(st.session_state.journal_ordres, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 4 : ACTUALITÉS
# ==============================================================================
elif menu_selection == "Actualités & Infos":
    st.title("Actualités Financières")
    RSS_FEEDS = {
        "🌍 Général (Les Echos)": "https://services.lesechos.fr/rss/une.xml",
        "📈 Bourse (Boursorama)": "https://www.boursorama.com/rss/actualites/economie",
        "🇺🇸 US Markets (CNBC)": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "💻 Tech & Crypto (CoinDesk)": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "🇪🇺 Économie (Euronews)": "https://fr.euronews.com/rss?format=xml&level=theme&name=business"
    }

    def fetch_news(feed_url):
        news_list = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(feed_url, headers=headers, timeout=5)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:8]:
                    news_list.append({ "title": entry.title, "link": entry.link, "published": entry.get("published", entry.get("updated", "")), "summary": entry.get("summary", "Lire l'article...")[:180] + "..." })
        except: pass
        return news_list

    source_choice = st.selectbox("Choisir une source :", list(RSS_FEEDS.keys()))
    if source_choice:
        with st.spinner("Chargement..."):
            news_items = fetch_news(RSS_FEEDS[source_choice])
        if news_items:
            st.markdown("---")
            col1, col2 = st.columns(2)
            for i, item in enumerate(news_items):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"""<div class="news-card"><div class="news-title">{item['title']}</div><div class="news-date">{item['published']}</div><div class="news-summary">{item['summary']}</div><br><a href="{item['link']}" target="_blank" class="news-link">Lire l'article complet →</a></div>""", unsafe_allow_html=True)
        else: st.warning("Aucune info disponible.")
