import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import feedparser
import requests
import yfinance as yf
import google.generativeai as genai
from datetime import datetime, timedelta
import time

# --- CONFIGURATION GLOBALE ---
st.set_page_config(
    page_title="Wealth Manager Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN SYSTEM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #1F1F1F; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    div[data-testid="metric-container"], .news-card { background-color: #161A25; border: 1px solid #252A38; padding: 15px; border-radius: 12px; }
    .stChatMessage { background-color: #161A25; border-radius: 10px; border: 1px solid #252A38; }
    .stButton>button { width: 100%; border: 1px solid #D4AF37; color: #D4AF37; background: transparent; }
    .stButton>button:hover { background-color: #D4AF37; color: #000000; }
    </style>
    """, unsafe_allow_html=True)

# --- SÉCURITÉ ---
if st.secrets.get("password"):
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        pwd = st.text_input("Access Key", type="password")
        if pwd == st.secrets["password"]: st.session_state.password_correct = True; st.rerun()
        else: st.stop()

# --- INITIALISATION ---
if 'journal_ordres' not in st.session_state:
    st.session_state.journal_ordres = pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Quantité', 'Prix Unitaire', 'Frais', 'Total'])
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour. Je suis Gemini Finance. Je suis connecté aux marchés. Posez-moi une question sur votre stratégie ou l'économie actuelle."}]

try: from constantes import CATALOGUE
except ImportError: CATALOGUE = {}

# --- FONCTIONS UTILITAIRES ---
@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = { "VIX": "^VIX", "US10Y": "^TNX", "Gold": "GC=F", "Oil": "CL=F", "EURUSD": "EURUSD=X" }
    try:
        data = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
        info = {}
        for k, t in tickers.items():
            try:
                # Gestion robuste des formats yfinance
                s = data[t] if isinstance(data.columns, pd.MultiIndex) else (data[t] if t in data else data.iloc[:,0])
                info[k] = (float(s.iloc[-1]), (float(s.iloc[-1])/float(s.iloc[-2])-1)*100)
            except: info[k] = (0.0, 0.0)
        return info
    except: return {}

@st.cache_data
def get_stock_data_optimized(tickers, start, end):
    if not tickers: return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        # Gestion des colonnes multiples ou simples
        if 'Adj Close' in df: return df['Adj Close']
        elif 'Close' in df: return df['Close']
        return df
    except: return pd.DataFrame()

def simulate_dca(ticker, monthly, years):
    data = get_stock_data_optimized([ticker], datetime.now()-timedelta(days=years*365), datetime.now())
    if data.empty: return None
    # Conversion explicite en DataFrame si Série
    if isinstance(data, pd.Series): data = data.to_frame(name=ticker)
    
    # Resample mensuel
    monthly_data = data.resample('M').last() 
    invested, shares, val_hist, inv_hist, dates = 0, 0, [], [], []
    
    # Gestion si plusieurs colonnes (prend la première)
    price_col = monthly_data.columns[0]
    
    for date, row in monthly_data.iterrows():
        price = row[price_col] if isinstance(row, pd.Series) else row
        if pd.isna(price): continue
        shares += monthly / price
        invested += monthly
        val_hist.append(shares * price)
        inv_hist.append(invested)
        dates.append(date)
    return pd.DataFrame({'Date': dates, 'Valeur': val_hist, 'Investi': inv_hist})

def get_readable_name(ticker):
    if not CATALOGUE: return ticker
    for cat in CATALOGUE.values():
        for name, code in cat.items():
            if code == ticker: return f"{ticker} ({name.split(' (')[0]})"
    return ticker

# --- IA ROBUSTE (TENTATIVE MULTIPLE) ---
def get_ai_response_gemini(user_prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "⚠️ Clé API manquante dans les secrets."
    
    genai.configure(api_key=api_key)
    
    macro = get_macro_data()
    macro_txt = ", ".join([f"{k}:{v[0]:.2f}" for k,v in macro.items()])
    
    # Prompt système
    sys_prompt = f"""Tu es un expert finance (Niveau ENS).
    Marché: {macro_txt}.
    Règles: Sois critique, analyse les risques, sois concis."""
    
    # Liste des modèles à tester par ordre de priorité
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']
    
    last_error = ""
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"{sys_prompt}\n\nQuestion: {user_prompt}")
            return response.text
        except Exception as e:
            last_error = str(e)
            continue # On passe au modèle suivant si celui-ci échoue
            
    return f"Erreur IA (Tous les modèles ont échoué). Détail : {last_error}"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### WEALTH MANAGER")
    menu = st.radio("NAV", ["Tableau de Bord", "Marchés", "Transactions", "Conseiller IA", "DCA", "Inflation", "Actualités"], label_visibility="collapsed")
    st.caption("v8.0 • Robust AI")

# --- PAGES ---
if menu == "Tableau de Bord":
    st.title("Synthèse")
    c1, c2, c3 = st.columns(3)
    la = c1.number_input("Liquidités", value=10000.0)
    bo = c2.number_input("Bourse", value=5000.0)
    cr = c3.number_input("Crypto", value=1000.0)
    tot = la + bo + cr
    st.metric("Total", f"{tot:,.0f} €")
    fig = px.pie(values=[la, bo, cr], names=['Cash', 'Bourse', 'Crypto'], hole=0.7, color_discrete_sequence=['#333333', '#D4AF37', '#FAFAFA'])
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Marchés":
    st.title("Marchés")
    # Affichage Macro
    macro = get_macro_data()
    cols = st.columns(len(macro))
    for i, (k, v) in enumerate(macro.items()): cols[i].metric(k, f"{v[0]:.2f}", f"{v[1]:+.1f}%")
    
    t = st.text_input("Ajout Ticker", "AAPL")
    if t:
        df = get_stock_data_optimized([t], datetime.now()-timedelta(days=365), datetime.now())
        if not df.empty: st.line_chart(df)

elif menu == "Transactions":
    st.title("Journal")
    c1, c2 = st.columns(2)
    with c1: st.download_button("💾 Save", st.session_state.journal_ordres.to_csv(index=False), "journal.csv")
    with c2: 
        up = st.file_uploader("📂 Load", type=['csv'], label_visibility="collapsed")
        if up: st.session_state.journal_ordres = pd.read_csv(up)
    
    with st.form("add"):
        c = st.columns(5)
        d = c[0].date_input("Date")
        ti = c[1].text_input("Ticker", "AAPL")
        s = c[2].selectbox("Sens", ["Achat", "Vente"])
        q = c[3].number_input("Qté", 1.0)
        p = c[4].number_input("Prix", 100.0)
        if st.form_submit_button("Ajouter"):
            n = pd.DataFrame([{'Date': d, 'Ticker': ti.upper(), 'Type': s, 'Quantité': q, 'Prix Unitaire': p, 'Frais': 0, 'Total': q*p}])
            st.session_state.journal_ordres = pd.concat([st.session_state.journal_ordres, n], ignore_index=True)
            st.rerun()
    st.dataframe(st.session_state.journal_ordres, use_container_width=True, hide_index=True)

elif menu == "Conseiller IA":
    st.title("Conseiller IA (Gemini)")
    for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
    
    if prompt := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                resp = get_ai_response_gemini(prompt)
                st.write(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

elif menu == "DCA":
    st.title("Simulateur DCA")
    c1, c2 = st.columns(2)
    ti = c1.text_input("Ticker", "SPY")
    mo = c2.number_input("Mensuel", 200)
    if st.button("Simuler"):
        res = simulate_dca(ti, mo, 10)
        if res is not None:
            st.metric("Valeur Finale", f"{res['Valeur'].iloc[-1]:,.0f} €")
            st.line_chart(res.set_index('Date')[['Valeur', 'Investi']])

elif menu == "Inflation":
    st.title("Calculateur Inflation")
    s = st.number_input("Somme", 10000)
    y = st.slider("Années", 1, 30, 20)
    i = st.slider("Inflation", 0.0, 10.0, 2.5)/100
    res = s / ((1+i)**y)
    st.metric("Pouvoir d'achat futur", f"{res:,.0f} €", f"-{s-res:,.0f} €", delta_color="inverse")

elif menu == "Actualités":
    st.title("News")
    feeds = ["https://services.lesechos.fr/rss/une.xml", "https://www.boursorama.com/rss/actualites/economie"]
    for f in feeds:
        d = feedparser.parse(f)
        for e in d.entries[:3]:
            st.markdown(f"**{e.title}**")
            st.caption(f"{e.get('published', '')} - [Lire]({e.link})")
            st.markdown("---")
