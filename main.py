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
    div[data-testid="metric-container"], .news-card, .macro-card, .lab-card {
        background-color: #161A25;
        border: 1px solid #252A38;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    div[data-testid="metric-container"]:hover, .news-card:hover, .lab-card:hover {
        border-color: #D4AF37;
    }
    
    /* NEWS STYLE */
    .news-title { font-size: 16px; font-weight: 600; color: #FAFAFA; margin-bottom: 5px; }
    .news-date { font-size: 12px; color: #D4AF37; margin-bottom: 10px; }
    .news-summary { font-size: 14px; color: #B0B0B0; }
    .news-link { color: #D4AF37; text-decoration: none; font-size: 14px; }

    /* CHATBOT STYLE */
    .stChatMessage { background-color: #161A25; border-radius: 10px; border: 1px solid #252A38; }
    [data-testid="stChatMessageContent"] { color: #FAFAFA; }
    [data-testid="stChatMessageAvatarUser"] { background-color: #D4AF37; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #000000; border: 1px solid #D4AF37; }

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
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour. Je suis Gemini Finance, connecté aux marchés en temps réel. Je peux analyser la conjoncture, évaluer des actifs et critiquer vos choix d'investissement. Quelle est votre question ?"}]

try: from constantes import CATALOGUE
except ImportError: CATALOGUE = {}

# --- FONCTIONS UTILITAIRES ---
def get_readable_name(ticker):
    if not CATALOGUE: return ticker
    for cat in CATALOGUE.values():
        for name, code in cat.items():
            if code == ticker: return f"{ticker} ({name.split(' (')[0]})"
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

@st.cache_data(ttl=86400)
def get_fundamentals(tickers):
    infos = []
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            dat = ticker_obj.info 
            sector = dat.get('sector', 'N/A')
            per = dat.get('trailingPE', None)
            div = dat.get('dividendYield', None)
            per_str = f"{per:.1f}" if per else "-"
            div_str = f"{div*100:.2f}%" if div else "-"
            infos.append({ "Actif": t, "Nom": dat.get('shortName', t), "Secteur": sector, "PER (Cherté)": per_str, "Dividende": div_str })
        except:
            infos.append({"Actif": t, "Nom": "-", "Secteur": "-", "PER (Cherté)": "-", "Dividende": "-"})
    return pd.DataFrame(infos)

@st.cache_data
def get_stock_data_optimized(tickers, start, end):
    if not tickers: return pd.DataFrame()
    try: df = yf.download(tickers, start=start, end=end, progress=False)
    except: return pd.DataFrame()
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

def simulate_dca(ticker, monthly_amount, years):
    start_date = datetime.now() - timedelta(days=years*365)
    data = get_stock_data_optimized([ticker], start_date, datetime.now())
    if data.empty: return None
    monthly_data = data.resample('M').last()
    total_invested = 0
    total_shares = 0
    history_val = []
    history_invest = []
    dates = []
    for date, price in monthly_data.itertuples():
        if pd.isna(price): continue
        shares_bought = monthly_amount / price
        total_shares += shares_bought
        total_invested += monthly_amount
        current_value = total_shares * price
        dates.append(date)
        history_val.append(current_value)
        history_invest.append(total_invested)
    return pd.DataFrame({'Date': dates, 'Valeur Portefeuille': history_val, 'Total Investi': history_invest})

# --- CONFIGURATION IA (GEMINI REAL) ---
def get_ai_response_gemini(user_prompt):
    # 1. Vérifier si la clé API est présente
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Erreur : Clé API Gemini manquante. Veuillez l'ajouter dans les Secrets de Streamlit."

    genai.configure(api_key=api_key)
    
    # 2. Récupérer le contexte du marché (RAG)
    macro = get_macro_data()
    macro_text = "Données Macro (Temps réel) : " + ", ".join([f"{k}: {v[0]:.2f} (Var: {v[1]:+.2f}%)" for k,v in macro.items()])
    
    # 3. Récupérer le contexte utilisateur
    portfolio_context = "L'utilisateur n'a pas encore saisi de transactions."
    if not st.session_state.journal_ordres.empty:
        tickers = st.session_state.journal_ordres['Ticker'].unique()
        portfolio_context = f"Portefeuille actuel de l'utilisateur : {', '.join(tickers)}."

    # 4. Construire le Prompt Expert
    system_instruction = f"""
    Tu es "Gemini Finance", un analyste financier expert et pragmatique pour un étudiant en Prépa D2 (Économie Gestion).
    
    CONTEXTE ACTUEL :
    {macro_text}
    {portfolio_context}
    
    TES MISSIONS :
    1. Analyse : Utilise les données macro (VIX, Taux, etc.) pour contextualiser ta réponse.
    2. Critique : Pour chaque actif ou stratégie mentionné, tu DOIS présenter les risques, les inconvénients et le scénario "Bear Case" (pessimiste). Ne sois jamais complaisant.
    3. Pédagogie : Explique les concepts (volatilité, bêta, corrélation) si nécessaire.
    4. Prudence : Rappelle toujours que les performances passées ne préjugent pas des futures.
    
    Réponds de manière structurée et professionnelle.
    """
    
    try:
        # --- FIX : UTILISATION DE GEMINI-PRO AU LIEU DE FLASH ---
        model = genai.GenerativeModel('gemini-pro') 
        full_prompt = f"{system_instruction}\n\nQuestion utilisateur : {user_prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Erreur de connexion à l'IA : {str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### WEALTH MANAGER")
    st.markdown("---")
    menu_selection = st.radio(
        "NAVIGATION", 
        ["Tableau de Bord", "Marchés & Analyse", "Transactions", "Conseiller IA", "Simulateur DCA", "Calculateur Inflation", "Actualités & Infos"], 
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("v7.1 • Gemini Pro Fix")

# ==============================================================================
# PAGES 
# ==============================================================================
if menu_selection == "Tableau de Bord":
    st.title("Synthèse & Rente Future")
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
        c_a, c_b, c_c = st.columns(3)
        apport = c_a.number_input("Apport/mois (€)", value=500.0)
        taux = c_b.slider("Rendement (%)", 0, 15, 7) / 100
        yield_target = c_c.slider("Yield (%)", 0, 8, 3) / 100
        annees = 20
        valeurs = [total_wealth]
        rentes = [total_wealth * yield_target / 12]
        for _ in range(annees):
            nouveau_cap = valeurs[-1] * (1 + taux) + (apport * 12)
            valeurs.append(nouveau_cap)
            rentes.append(nouveau_cap * yield_target / 12)
        df_proj = pd.DataFrame({'Année': range(annees + 1), 'Capital': valeurs, 'Rente Mensuelle': rentes})
        tab_cap, tab_rente = st.tabs(["💰 CAPITAL", "🏖️ RENTE"])
        with tab_cap:
            fig_area = px.area(df_proj, x='Année', y='Capital')
            fig_area.update_traces(line_color='#D4AF37', fillcolor='rgba(212, 175, 55, 0.1)')
            st.plotly_chart(style_plotly(fig_area), use_container_width=True)
        with tab_rente:
            fig_bar = px.bar(df_proj, x='Année', y='Rente Mensuelle')
            fig_bar.update_traces(marker_color='#FAFAFA')
            st.plotly_chart(style_plotly(fig_bar), use_container_width=True)

elif menu_selection == "Marchés & Analyse":
    st.markdown("### INDICATEURS MACRO")
    macro_data = get_macro_data()
    if macro_data:
        cols_macro = st.columns(len(macro_data))
        for i, (name, (val, delta)) in enumerate(macro_data.items()):
            cols_macro[i].metric(name, f"{val:.2f}", f"{delta:+.2f}%")
    st.markdown("---")
    st.title("Analyse de Portefeuille")
    col_input, col_period = st.columns([3, 1])
    with col_input:
        selected_tickers = []
        if CATALOGUE:
            for cat, assets in CATALOGUE.items():
                sel = st.multiselect(f"{cat}", list(assets.keys()))
                for nom in sel: selected_tickers.append(assets[nom])
        manual = st.text_input("Ajout Manuel", placeholder="Ex: AI.PA")
        if manual: selected_tickers.append(manual.upper())
        tickers = list(set(selected_tickers))
    with col_period:
        start_date = st.date_input("Depuis", value=datetime.now() - timedelta(days=365*2))
    if tickers:
        st.markdown("#### 🔎 FONDAMENTAUX")
        st.dataframe(get_fundamentals(tickers), use_container_width=True, hide_index=True)
        with st.spinner('Calculs...'):
            df_prices = get_stock_data_optimized(tickers, start_date, datetime.now())
        if not df_prices.empty:
            tab_alloc, tab_optim = st.tabs(["📊 ALLOCATION", "🧠 MARKOWITZ"])
            with tab_alloc:
                cols = st.columns(4)
                weights = []
                found = df_prices.columns.tolist()
                for i, t in enumerate(found):
                    weights.append(st.number_input(get_readable_name(t), 0.0, 1.0, 1.0/len(found), 0.05, key=f"w_{t}"))
                df_norm = (df_prices / df_prices.iloc[0]) * 100
                returns = df_prices.pct_change().dropna()
                portf_ret = returns.dot(weights)
                portf_cum = (1 + portf_ret).cumprod() * 100
                df_final = df_norm.copy()
                df_final['PORTFOLIO'] = portf_cum.fillna(100)
                try:
                    bench = get_stock_data_optimized(['SPY'], start_date, datetime.now())
                    if not bench.empty: df_final['S&P 500'] = (bench / bench.iloc[0]) * 100
                except: pass
                colors = ["#D4AF37" if c == "PORTFOLIO" else "#FAFAFA" if c == "S&P 500" else "#333333" for c in df_final.columns]
                st.line_chart(df_final, color=colors)
                c1, c2, c3 = st.columns(3)
                var_95 = np.percentile(portf_ret, 5)
                from logic.metrics import calculate_key_metrics, get_correlation_matrix
                with c1: st.dataframe(calculate_key_metrics(pd.DataFrame({'Portfolio': df_final['PORTFOLIO']})).T.style.format("{:.2f}"))
                with c2: st.metric("VaR (95%)", f"{var_95:.2%}")
                with c3: st.dataframe(get_correlation_matrix(df_prices).style.background_gradient(cmap='cividis', axis=None).format("{:.2f}"))
            with tab_optim:
                if st.button("Lancer Optimisation"):
                    results, weights_record = run_monte_carlo_simulation(df_prices)
                    max_idx = np.argmax(results[2])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results[1,:], y=results[0,:], mode='markers', marker=dict(color=results[2,:], colorscale='Cividis', size=5)))
                    fig.add_trace(go.Scatter(x=[results[1, max_idx]], y=[results[0, max_idx]], mode='markers', marker=dict(color='#D4AF37', size=15)))
                    st.plotly_chart(fig, use_container_width=True)
                    full_names = [get_readable_name(t) for t in df_prices.columns]
                    opt_df = pd.DataFrame({'Actif': full_names, 'Poids Idéal': weights_record[max_idx]})
                    st.dataframe(opt_df.T)

elif menu_selection == "Transactions":
    st.title("Journal")
    c1, c2 = st.columns(2)
    with c1: st.download_button("💾 Sauvegarder CSV", data=st.session_state.journal_ordres.to_csv(index=False).encode('utf-8'), file_name="journal.csv", mime="text/csv")
    with c2: 
        up = st.file_uploader("📂 Charger CSV", type=['csv'], label_visibility="collapsed")
        if up: st.session_state.journal_ordres = pd.read_csv(up); st.success("Chargé !")
    st.markdown("---")
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
    st.dataframe(st.session_state.journal_ordres, use_container_width=True, hide_index=True)

elif menu_selection == "Conseiller IA":
    st.title("Conseiller Patrimonial (Gemini)")
    st.markdown("Discussion instantanée avec votre assistant expert.")
    
    if not st.secrets.get("GEMINI_API_KEY"):
        st.error("⚠️ Clé API manquante. Ajoutez GEMINI_API_KEY dans les secrets.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question (ex: 'Analyse mon portefeuille')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Analyse des marchés en cours..."):
                if st.secrets.get("GEMINI_API_KEY"):
                    ai_reply = get_ai_response_gemini(prompt)
                else:
                    ai_reply = "Veuillez configurer votre clé API Gemini pour activer l'intelligence."
            
            for chunk in ai_reply.split(" "):
                full_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif menu_selection == "Simulateur DCA":
    st.title("Simulateur DCA")
    c1, c2, c3 = st.columns(3)
    sim_ticker = c1.text_input("Actif", value="SPY")
    sim_amount = c2.number_input("Mensuel (€)", value=200)
    sim_years = c3.slider("Années", 3, 20, 10)
    if st.button("Simuler"):
        df_dca = simulate_dca(sim_ticker, sim_amount, sim_years)
        if df_dca is not None:
            final_val = df_dca['Valeur Portefeuille'].iloc[-1]
            total_inv = df_dca['Total Investi'].iloc[-1]
            perf = ((final_val - total_inv) / total_inv) * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Investi", f"{total_inv:,.0f} €")
            c2.metric("Final", f"{final_val:,.0f} €", f"{perf:+.2f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_dca['Date'], y=df_dca['Total Investi'], fill='tozeroy', name='Investi'))
            fig.add_trace(go.Scatter(x=df_dca['Date'], y=df_dca['Valeur Portefeuille'], fill='tonexty', name='Valeur'))
            st.plotly_chart(fig, use_container_width=True)

elif menu_selection == "Calculateur Inflation":
    st.title("Calculateur Inflation")
    ci1, ci2, ci3 = st.columns(3)
    somme = ci1.number_input("Somme (€)", value=10000)
    hz = ci2.slider("Années", 1, 40, 20)
    inf = ci3.slider("Inflation (%)", 0.0, 10.0, 2.5) / 100
    futur = somme / ((1 + inf) ** hz)
    perte = somme - futur
    st.metric("Pouvoir d'achat futur", f"{futur:,.0f} €", f"-{perte:,.0f} €", delta_color="inverse")
    vals = [somme / ((1 + inf) ** y) for y in range(hz + 1)]
    fig = px.area(x=range(hz + 1), y=vals, labels={'x':'Années', 'y':'Valeur Réelle'})
    fig.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig, use_container_width=True)

elif menu_selection == "Actualités & Infos":
    st.title("Actualités")
    RSS_FEEDS = { "🌍 Les Echos": "https://services.lesechos.fr/rss/une.xml", "📈 Boursorama": "https://www.boursorama.com/rss/actualites/economie", "🇺🇸 CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664" }
    def fetch_news(url):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                f = feedparser.parse(r.content)
                return [{"title": e.title, "link": e.link, "date": e.get("published", ""), "summary": e.get("summary", "")[:180]+"..."} for e in f.entries[:8]]
        except: pass
        return []
    
    src = st.selectbox("Source", list(RSS_FEEDS.keys()))
    if src:
        items = fetch_news(RSS_FEEDS[src])
        if items:
            c1, c2 = st.columns(2)
            for i, it in enumerate(items):
                with c1 if i % 2 == 0 else c2:
                    st.markdown(f"""<div class="news-card"><div class="news-title">{it['title']}</div><div class="news-date">{it['date']}</div><div class="news-summary">{it['summary']}</div><a href="{it['link']}" target="_blank" class="news-link">Lire →</a></div>""", unsafe_allow_html=True)
