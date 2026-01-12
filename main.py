import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- IMPORT MODULES LOCAUX ---
from data.loader import get_stock_data
from logic.metrics import calculate_key_metrics, get_correlation_matrix

try:
    from constantes import CATALOGUE
except ImportError:
    CATALOGUE = {}

# --- 1. CONFIGURATION GLOBALE ---
st.set_page_config(
    page_title="Wealth Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DESIGN SYSTEM "PURE & GOLD" (CSS AVANCÉ) ---
st.markdown("""
    <style>
    /* Import Police Moderne (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    /* RESET GLOBAL */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; /* Noir Profond */
        color: #FAFAFA;
    }

    /* --- SIDEBAR (MENU GAUCHE) --- */
    section[data-testid="stSidebar"] {
        background-color: #000000; /* Noir pur */
        border-right: 1px solid #1F1F1F;
    }
    
    /* --- GESTION DE LA FLÈCHE (HEADER) --- */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Bouton d'ouverture/fermeture du menu */
    button[kind="header"] {
        background-color: transparent;
        color: #D4AF37 !important; /* Flèche en OR */
    }
    [data-testid="stSidebarCollapsedControl"] {
        color: #D4AF37 !important;
        background-color: transparent;
        display: block !important;
    }

    /* --- TITRES --- */
    h1, h2, h3 {
        color: #D4AF37 !important; /* OR */
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    h4, h5 {
        color: #A0A0A0 !important;
        font-weight: 400;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    /* --- CARDS (LES BLOCS ARRONDIS) --- */
    div[data-testid="metric-container"] {
        background-color: #161A25;
        border: 1px solid #252A38;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #D4AF37;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #888;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #FAFAFA !important;
        font-weight: 600;
    }

    /* --- BOUTONS --- */
    .stButton>button {
        width: 100%;
        background-color: transparent;
        border: 1px solid #D4AF37;
        color: #D4AF37;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #000000;
        border: 1px solid #D4AF37;
    }

    /* --- INPUTS --- */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #161A25;
        border: 1px solid #252A38;
        color: #FAFAFA;
        border-radius: 8px;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SÉCURITÉ ---
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
    with c2:
        st.text_input("Access Key", type="password", on_change=password_entered, key="password")
    return False

if not check_password(): st.stop()

# --- 4. INITIALISATION MÉMOIRE ---
if 'journal_ordres' not in st.session_state:
    st.session_state.journal_ordres = pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Quantité', 'Prix Unitaire', 'Frais', 'Total'])

# --- 5. FONCTION STYLE GRAPHIQUE ---
def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#A0A0A0", family="Inter"),
        margin=dict(t=30, l=0, r=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#252A38', zeroline=False),
        colorway=["#D4AF37", "#FFFFFF", "#808080", "#555555"]
    )
    return fig

# ==============================================================================
# MENU LATÉRAL (SIDEBAR NAVIGATION)
# ==============================================================================
with st.sidebar:
    # TITRE MODIFIÉ ICI (Suppression de l'étoile)
    st.markdown("### WEALTH MANAGER")
    st.markdown("---")
    
    menu_selection = st.radio(
        "NAVIGATION",
        ["Tableau de Bord", "Marchés & Analyse", "Transactions"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### RECHERCHE RAPIDE")
    search_query = st.text_input("Rechercher un actif...", placeholder="AAPL, BTC...")
    st.markdown("---")
    st.caption("v1.4.0 • Secure Mode")

# ==============================================================================
# PAGE 1 : MON PATRIMOINE (TABLEAU DE BORD)
# ==============================================================================
if menu_selection == "Tableau de Bord":
    st.title("Synthèse Patrimoniale")
    st.markdown("#### VUE D'ENSEMBLE")
    
    with st.expander("📝 Mettre à jour mes soldes", expanded=True):
        c1, c2, c3 = st.columns(3)
        livret_a = c1.number_input("Liquidités (€)", value=10000.0, step=100.0)
        bourse = c2.number_input("Bourse (€)", value=5000.0, step=100.0)
        crypto = c3.number_input("Crypto (€)", value=1000.0, step=100.0)
    
    total_wealth = livret_a + bourse + crypto
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Valeur Nette Totale", f"{total_wealth:,.0f} €", "+2.4%")
    col2.metric("Liquidités", f"{livret_a:,.0f} €", "Sécurisé")
    col3.metric("Actifs Risqués", f"{bourse+crypto:,.0f} €", f"{((bourse+crypto)/total_wealth)*100:.1f}% du total")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_chart1, c_chart2 = st.columns([1, 2])
    
    with c_chart1:
        st.markdown("#### RÉPARTITION")
        if total_wealth > 0:
            fig = px.pie(
                values=[livret_a, bourse, crypto], 
                names=['Liquidités', 'Bourse', 'Crypto'], 
                hole=0.7,
                color_discrete_sequence=['#333333', '#D4AF37', '#FAFAFA']
            )
            fig.update_traces(textinfo='none', hovertemplate='%{label}: <b>%{value:,.0f} €</b>')
            fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            fig = style_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)

    with c_chart2:
        st.markdown("#### PROJECTION FUTURE")
        col_sim_a, col_sim_b = st.columns(2)
        apport = col_sim_a.number_input("Apport/mois", value=500.0)
        taux = col_sim_b.slider("Rendement (%)", 0, 15, 6) / 100
        
        valeurs = [total_wealth]
        for _ in range(15): valeurs.append(valeurs[-1] * (1 + taux) + (apport * 12))
        
        df_proj = pd.DataFrame({'Année': range(16), 'Capital': valeurs})
        fig_area = px.area(df_proj, x='Année', y='Capital')
        fig_area.update_traces(line_color='#D4AF37', fillcolor='rgba(212, 175, 55, 0.1)')
        fig_area = style_plotly(fig_area)
        st.plotly_chart(fig_area, use_container_width=True)

# ==============================================================================
# PAGE 2 : MARCHÉS & ANALYSE
# ==============================================================================
elif menu_selection == "Marchés & Analyse":
    st.title("Analyse de Marché")
    
    default_tickers = []
    if search_query: default_tickers = [search_query.upper().strip()]
    
    col_input, col_period = st.columns([3, 1])
    with col_input:
        st.markdown("#### SÉLECTION D'ACTIFS")
        selected_tickers = default_tickers.copy()
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
        st.markdown("---")
        with st.spinner('Connexion aux données de marché...'):
            df_prices = get_stock_data(tickers, start_date, datetime.now())
        
        if not df_prices.empty:
            st.markdown("#### 1. ALLOCATION STRATÉGIQUE")
            cols = st.columns(4)
            weights = []
            found_tickers = df_prices.columns.tolist()
            for i, t in enumerate(found_tickers):
                with cols[i % 4]:
                    w = st.number_input(f"{t}", 0.0, 1.0, 1.0/len(found_tickers), 0.05, key=f"w_{t}")
                    weights.append(w)
            
            # Calculs
            df_norm = (df_prices / df_prices.iloc[0]) * 100
            returns = df_prices.pct_change().dropna()
            portf = returns.dot(weights)
            portf_cum = (1 + portf).cumprod() * 100
            
            df_final = df_norm.copy()
            df_final['PORTFOLIO'] = portf_cum
            df_final['PORTFOLIO'] = df_final['PORTFOLIO'].fillna(100)
            
            st.markdown("#### 2. PERFORMANCE COMPARÉE")
            
            colors = []
            for c in df_final.columns:
                if c == "PORTFOLIO": colors.append("#D4AF37")
                else: colors.append("#444444")
            
            st.line_chart(df_final, color=colors)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### INDICATEURS DE RISQUE")
                metrics = calculate_key_metrics(pd.DataFrame({'Portfolio': df_final['PORTFOLIO']}))
                st.dataframe(metrics.T.style.format("{:.2f}"))
            with col_m2:
                st.markdown("#### DIVERSIFICATION (CORRÉLATION)")
                corr = get_correlation_matrix(df_prices)
                st.dataframe(corr.style.background_gradient(cmap='cividis', axis=None).format("{:.2f}"))

# ==============================================================================
# PAGE 3 : TRANSACTIONS (JOURNAL)
# ==============================================================================
elif menu_selection == "Transactions":
    st.title("Journal des Transactions")
    
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        csv = st.session_state.journal_ordres.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Sauvegarder CSV", data=csv, file_name="journal.csv", mime="text/csv")
    with c_btn2:
        up = st.file_uploader("📂 Charger CSV", type=['csv'], label_visibility="collapsed")
        if up:
            try:
                st.session_state.journal_ordres = pd.read_csv(up)
                st.success("Chargé !")
            except: pass

    st.markdown("---")
    
    st.markdown("#### NOUVEL ORDRE")
    with st.form("new_order"):
        c1, c2, c3, c4, c5 = st.columns(5)
        d = c1.date_input("Date")
        t = c2.text_input("Ticker", placeholder="AAPL")
        sens = c3.selectbox("Sens", ["Achat", "Vente"])
        q = c4.number_input("Qté", min_value=0.01)
        p = c5.number_input("Prix Unit.", min_value=0.01)
        
        if st.form_submit_button("VALIDER LA TRANSACTION"):
            new = {'Date': d, 'Ticker': t.upper(), 'Type': sens, 'Quantité': q, 'Prix Unitaire': p, 'Frais': 0, 'Total': q*p}
            st.session_state.journal_ordres = pd.concat([st.session_state.journal_ordres, pd.DataFrame([new])], ignore_index=True)
            st.rerun()

    st.markdown("#### HISTORIQUE")
    st.dataframe(
        st.session_state.journal_ordres, 
        use_container_width=True,
        hide_index=True
    )
    