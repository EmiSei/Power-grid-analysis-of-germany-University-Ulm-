# Project Work for University Ulm: Institute for Energy Conversion and Storage
import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pypsa
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Power Grid Analysis", layout="wide", page_icon="⚡")
st.title("⚡ Interactive Power Grid Analysis (Germany)")

API_KEY = os.environ.get("ENTSOE_API_KEY", '3bc1d876-8b15-4cec-a82a-64850e7427ed')

# ==========================================
# 2. HELPER FUNCTIONS & TRANSLATIONS
# ==========================================
def clean_german_num(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    cleaned = str(val).replace(',', '.')
    try: return float(''.join(c for c in cleaned if c.isdigit() or c == '.'))
    except: return 0.0

translation = {
    "Solare Strahlungsenergie": "Solar", "SolareStrahlungsenergie": "Solar",
    "Windenergie (Onshore-Anlage)": "Wind Onshore", "Wind (onshore)": "Wind Onshore",
    "Windenergie (Offshore-Anlage)": "Wind Offshore", "Wind (offshore)": "Wind Offshore",
    "Wasserkraft": "Hydro", "Wasser": "Hydro",
    "Erdgas": "Gas", "Braunkohle": "Lignite", "Steinkohle": "Hard Coal",
    "Biomasse": "Biomass", "Speicher": "Battery", "Batteriespeicher": "Battery",
    "Pumpspeicher": "Pumped Hydro",
    "Kernenergie": "Nuclear", "Kernkraft": "Nuclear", 
    "Öl": "Oil", "Mineraloelprodukte": "Oil",
    "Abfall": "Waste"
}

color_map = {
    "Lignite": "#8B4513", "Hard Coal": "#000000", "Gas": "#FFA500",
    "Wind Onshore": "#00BFFF", "Wind Offshore": "#008B8B", "Nuclear": "#FF0000",
    "Solar": "#FFD700", "Hydro": "#4169E1", "Biomass": "#228B22",
    "Battery": "#AA00FF", "Pumped Hydro": "#4B0082", "Waste": "#7f8c8d",
    "Household (Low)": "#19D3F3", "Industry (High)": "#FFA15A", 
    "SuedLink (DC)": "#E74C3C", "AC Grid": "#bdc3c7"
}

# ==========================================
# 3. DOWNLOAD & ENTSO-E DATA
# ==========================================
@st.cache_data(show_spinner=False)
def download_kraftwerksliste():
    CSV_DOWNLOAD_URL = 'https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/Versorgungssicherheit/Erzeugungskapazitaeten/Kraftwerksliste/_DL/Kraftwerksliste_CSV.csv?__blob=publicationFile&v=8'
    try:
        r = requests.get(CSV_DOWNLOAD_URL, timeout=10)
        r.raise_for_status()
        with open('./downloaded_kraftwerksliste.csv', 'wb') as f: f.write(r.content)
        return './downloaded_kraftwerksliste.csv'
    except: return None

def fetch_entsoe_data(api_key, start, end):
    try:
        client = EntsoePandasClient(api_key=api_key)
        country_code = 'DE'

        df_gen = client.query_generation(country_code, start=start, end=end)
        if not df_gen.empty:
            df_gen = df_gen.loc[:, (slice(None), 'Actual Aggregated')]
            df_gen.columns = df_gen.columns.droplevel(1)
        
        df_final = pd.DataFrame()
        for col in df_gen.columns:
            name_lower = col.lower()
            if 'wind' in name_lower: new_name = 'Wind'
            elif 'solar' in name_lower: new_name = 'Solar'
            elif 'hard coal' in name_lower: new_name = 'Hard Coal'
            elif 'brown coal' in name_lower or 'lignite' in name_lower: new_name = 'Brown Coal'
            elif 'gas' in name_lower: new_name = 'Natural Gas'
            elif 'oil' in name_lower: new_name = 'Oil'
            elif 'pumped storage' in name_lower: new_name = 'Pumped Storage'
            elif 'battery' in name_lower: new_name = 'Battery'
            elif 'hydro' in name_lower: new_name = 'Hydro'
            elif 'nuclear' in name_lower: new_name = 'Nuclear'
            elif 'biomass' in name_lower: new_name = 'Biomass'
            elif 'waste' in name_lower: new_name = 'Waste'
            else: new_name = col
            
            if new_name not in df_final.columns: df_final[new_name] = df_gen[col]
            else: df_final[new_name] += df_gen[col]

        df_load_raw = client.query_load(country_code, start=start, end=end)
        if isinstance(df_load_raw, pd.Series): df_load = df_load_raw.to_frame(name='Actual Load')
        else: df_load = df_load_raw.rename(columns={df_load_raw.columns[0]: 'Actual Load'})
        
        df_final = df_final.join(df_load, how='inner').fillna(0)
        df_final.index = pd.to_datetime(df_final.index)
        if df_final.index.tz is not None: df_final.index = df_final.index.tz_convert('UTC')
        else: df_final.index = df_final.index.tz_localize('UTC')

        return df_final
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_live_entsoe_data(api_key):
    end = pd.Timestamp.now(tz='Europe/Berlin')
    start = end - timedelta(days=3)
    return fetch_entsoe_data(api_key, start, end)

@st.cache_data(show_spinner=False)
def load_winter_entsoe_data(api_key):
    start = pd.Timestamp('2024-01-15', tz='Europe/Berlin')
    end = pd.Timestamp('2024-01-18', tz='Europe/Berlin')
    return fetch_entsoe_data(api_key, start, end)

@st.cache_data(show_spinner=False)
def load_summer_entsoe_data(api_key):
    start = pd.Timestamp('2024-06-15', tz='Europe/Berlin')
    end = pd.Timestamp('2024-06-18', tz='Europe/Berlin')
    return fetch_entsoe_data(api_key, start, end)

# ==========================================
# 4. BUILD PYPSA NETWORK
# ==========================================
@st.cache_data(show_spinner=False) 
def build_network(file_path):
    n = pypsa.Network()
    n.set_snapshots(range(24))
    error_msg = None

    nodes = {
        "Hamburg": (9.99, 53.55), "Bremen": (8.80, 53.07), "Berlin": (13.40, 52.52), 
        "Hannover": (9.73, 52.37), "Leipzig": (12.37, 51.33), "Dresden": (13.73, 51.05),
        "Köln": (6.95, 50.93), "Frankfurt": (8.68, 50.11), "Ludwigshafen": (8.44, 49.48), 
        "Freiburg": (7.84, 47.99), "Nürnberg": (11.07, 49.45), "Ulm": (9.98, 48.40),
        "München": (11.57, 48.13), "Stuttgart": (9.18, 48.78), "Rostock": (12.13, 54.09), 
        "Magdeburg": (11.62, 52.13)
    }
    for name, (lon, lat) in nodes.items(): n.add("Bus", name=name, v_nom=380, x=lon, y=lat)

    lines = [
        ("Hamburg", "Bremen"), ("Bremen", "Köln"), ("Köln", "Frankfurt"),
        ("Hamburg", "Hannover"), ("Hannover", "Berlin"), ("Berlin", "Dresden"),
        ("Dresden", "Leipzig"), ("Leipzig", "Nürnberg"), ("Nürnberg", "München"),
        ("Frankfurt", "Ludwigshafen"), ("Ludwigshafen", "Freiburg"),
        ("Freiburg", "Ulm"), ("Frankfurt", "Nürnberg"), ("Hannover", "Frankfurt"),
        ("Ulm", "München"), ("Berlin", "Magdeburg"), ("Magdeburg", "Hannover"), 
        ("Hamburg", "Rostock"), ("Rostock", "Berlin"), ("Stuttgart", "Ulm"), 
        ("Stuttgart", "Freiburg"), ("Frankfurt", "Stuttgart")
    ]
    for b0, b1 in lines: n.add("Line", name=f"Line_{b0}_{b1}", bus0=b0, bus1=b1, type="Al/St 240/40 4-bundle 380", length=150)

    hh_profile = (15000 + 10000 * np.sin(np.pi * (np.arange(24) - 6) / 12) + 4000 * np.cos(np.pi * np.arange(24) / 6)) * 2.5
    ind_profile = np.full(24, 25000) * 1.8

    hh_shares = {"Hamburg": 0.08, "Bremen": 0.03, "Berlin": 0.14, "Hannover": 0.06, "Leipzig": 0.06, "Dresden": 0.07, "Köln": 0.21, "Frankfurt": 0.09, "Ludwigshafen": 0.03, "Freiburg": 0.05, "Nürnberg": 0.06, "Ulm": 0.03, "München": 0.14, "Stuttgart": 0.09, "Rostock": 0.04, "Magdeburg": 0.04}
    ind_shares = {"Hamburg": 0.08, "Bremen": 0.04, "Berlin": 0.04, "Hannover": 0.06, "Leipzig": 0.06, "Dresden": 0.04, "Köln": 0.31, "Frankfurt": 0.06, "Ludwigshafen": 0.14, "Freiburg": 0.05, "Nürnberg": 0.04, "Ulm": 0.05, "München": 0.03, "Stuttgart": 0.09, "Rostock": 0.03, "Magdeburg": 0.04}

    for node in nodes.keys():
        n.add("Load", name=f"Load_HH_{node}", bus=node, p_set=hh_profile * hh_shares[node])
        n.add("Load", name=f"Load_IND_{node}", bus=node, p_set=ind_profile * ind_shares[node])
        n.add("StorageUnit", name=f"Storage_{node}", bus=node, p_nom=1500, max_hours=4)

    if file_path is not None:
        try:
            header_idx = 0
            with open(file_path, 'r', encoding='latin-1') as f:
                for i, line in enumerate(f):
                    if 'Anlagenbetreiber' in line and 'Bundesland' in line:
                        header_idx = i
                        break
            
            df = pd.read_csv(file_path, sep=';', encoding='latin-1', skiprows=header_idx, engine='python', on_bad_lines='skip')
            
            col_bland = next((c for c in df.columns if 'Bundesland' in c), None)
            col_energie = next((c for c in df.columns if 'Energietraeger' in c or 'Energieträger' in c), None)
            p_col = next((c for c in df.columns if 'Nettonennleistung' in c and 'MW' in c), None)
            
            if not all([col_bland, col_energie, p_col]):
                raise ValueError("Critical columns could not be found in the dataset.")

            clean_df = df.copy()
            clean_df[p_col] = clean_df[p_col].apply(clean_german_num)
            clean_df[col_energie] = clean_df[col_energie].astype(str)
            
            if not clean_df[col_energie].str.contains('Pumpspeicher', case=False).any():
                mask_ps = clean_df[col_energie].str.contains('Speicher', case=False, na=False)
                clean_df.loc[mask_ps, col_energie] = 'Pumpspeicher'

            exclude_carriers = ['andere Gase', 'nicht biogener Abfall', 'Grubengas', 'Druck aus Gasleitungen', 'Waerme', 'Wärme', 'Geothermie', 'Klaerschlamm', 'Klärschlamm', 'Druck aus Wasserleitungen', 'Wasserstoff', 'Sonstige Energietraeger (nicht erneuerbar)']
            filtered_df = clean_df[~clean_df[col_energie].isin(exclude_carriers)].copy()

            bundesland_to_bus = {
                'Schleswig-Holstein': 'Hamburg', 'SchleswigHolstein': 'Hamburg',
                'Hamburg': 'Hamburg', 
                'Bremen': 'Bremen', 
                'Niedersachsen': 'Hannover',
                'Nordrhein-Westfalen': 'Köln', 'NordrheinWestfalen': 'Köln',
                'Hessen': 'Frankfurt', 
                'Rheinland-Pfalz': 'Ludwigshafen', 'RheinlandPfalz': 'Ludwigshafen',
                'Saarland': 'Ludwigshafen',
                'Baden-Württemberg': 'Stuttgart', 'BadenWuerttemberg': 'Stuttgart',
                'Bayern': 'München', 
                'Berlin': 'Berlin', 
                'Brandenburg': 'Berlin',
                'Mecklenburg-Vorpommern': 'Rostock', 'MecklenburgVorpommern': 'Rostock',
                'Sachsen': 'Dresden', 
                'Sachsen-Anhalt': 'Magdeburg', 'SachsenAnhalt': 'Magdeburg',
                'Thüringen': 'Leipzig', 'Thueringen': 'Leipzig',
                'Nordsee': 'Bremen', 'Ostsee': 'Rostock'
            }

            state_mix = filtered_df.groupby([col_bland, col_energie])[p_col].sum().reset_index()
            
            for _, row in state_mix.iterrows():
                bland = row[col_bland]
                if pd.isna(bland): continue
                
                target_bus = bundesland_to_bus.get(bland)
                leistung = row[p_col]
                carrier = row[col_energie]
                
                if target_bus in nodes and leistung > 0:
                    n.add("Generator", name=f"Gen_{bland}_{carrier}", bus=target_bus, p_nom=leistung, carrier=carrier)
                    
        except Exception as e: 
            error_msg = f"CSV Parsing Error: {str(e)}"
    
    return n, nodes, hh_profile, ind_profile, error_msg

# ==========================================
# 5. INITIALIZATION
# ==========================================
st.info("⏳ Loading data and initializing network... Please wait.")
kraftwerks_file = download_kraftwerksliste()
network, nodes_dict, hh_profile_base, ind_profile_base, csv_error = build_network(kraftwerks_file)

if csv_error:
    st.error(f"❌ {csv_error}")

df_entsoe_live = load_live_entsoe_data(API_KEY)
df_entsoe_winter = load_winter_entsoe_data(API_KEY)
df_entsoe_summer = load_summer_entsoe_data(API_KEY)
if not csv_error:
    st.success("✅ Network and all ENTSO-E datasets (Live, Winter, Summer) loaded!")

# ==========================================
# 6. REUSABLE SIMULATION UI FUNCTION
# ==========================================
def render_simulation_tab(df_source, tab_desc, prefix_key):
    st.subheader(f"🎛️ Custom Energy Transition Simulation ({tab_desc})")
    
    if df_source.empty or 'Actual Load' not in df_source.columns:
        st.warning(f"No data available for this period ({tab_desc}). Please check your API Key or timeframe.")
        return

    st.markdown("Determine the exact absolute storage capacities required to balance the historical residual load curve.")
    
    col1, col2, col3 = st.columns(3)
    with col1: scale_wind = st.slider("🌬️ Wind Power (Factor)", 0.0, 3.0, 1.0, 0.1, key=f"{prefix_key}_wind")
    with col2: scale_solar = st.slider("☀️ Solar Power (Factor)", 0.0, 3.0, 1.0, 0.1, key=f"{prefix_key}_solar")
    with col3: scale_fossil = st.slider("🏭 Fossil Fuels (Factor)", 0.0, 2.0, 1.0, 0.1, key=f"{prefix_key}_fossil")
    
    col4, col5 = st.columns(2)
    with col4: max_pumped_mw = st.slider("💧 Total Target Pumped Hydro (MW)", 0, 30000, 6500, 500, key=f"{prefix_key}_pumped")
    with col5: max_battery_mw = st.slider("🔋 Total Target Battery (MW)", 0, 60000, 5000, 500, key=f"{prefix_key}_batt")

    df_sim = df_source.copy()
    
    if 'Wind' in df_sim.columns: df_sim['Wind'] *= scale_wind
    if 'Solar' in df_sim.columns: df_sim['Solar'] *= scale_solar
    for col in ['Hard Coal', 'Brown Coal', 'Natural Gas', 'Oil']:
        if col in df_sim.columns: df_sim[col] *= scale_fossil
        
    re_cols = [c for c in ['Wind', 'Solar', 'Hydro', 'Biomass'] if c in df_sim.columns]
    df_sim['Total Renewable Gen'] = df_sim[re_cols].sum(axis=1)
    df_sim['Residual Load'] = df_sim['Actual Load'] - df_sim['Total Renewable Gen']
    
    df_sim['Simulated Pumped Hydro'] = df_sim['Residual Load'].apply(lambda x: min(max_pumped_mw, max(0, x)) if x > 0 else max(-max_pumped_mw, x))
    df_sim['Simulated Battery Storage'] = (df_sim['Residual Load'] - df_sim['Simulated Pumped Hydro']).apply(lambda x: min(max_battery_mw, max(0, x)) if x > 0 else max(-max_battery_mw, x))

    fig_sim = go.Figure()
    
    for col in [c for c in df_sim.columns if c not in ['Actual Load', 'Demand: Households', 'Demand: Industry', 'Total Renewable Gen', 'Residual Load', 'Simulated Pumped Hydro', 'Simulated Battery Storage', 'Pumped Storage', 'Battery']]:
        fig_sim.add_trace(go.Scatter(x=df_sim.index, y=df_sim[col], mode='lines', stackgroup='one', name=col))
        
    fig_sim.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Simulated Pumped Hydro'], mode='lines', name='Dispatched Pumped Hydro (Sim)', line=dict(color=color_map["Pumped Hydro"], width=2, dash='dash')))
    fig_sim.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Simulated Battery Storage'], mode='lines', name='Dispatched Battery Storage (Sim)', line=dict(color=color_map["Battery"], width=2, dash='dot')))
    fig_sim.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Actual Load'], mode='lines', name='ENTSO-E TOTAL DEMAND', line=dict(color='white', width=4)))
    
    fig_sim.update_layout(template='plotly_dark', hovermode='x unified', height=550, yaxis_title="Power (MW)", xaxis_title="Time (UTC)")
    
    st.plotly_chart(fig_sim, width='stretch')
    
    st.markdown("### 📊 Storage Adequacy Check")
    unmet_demand_peak = (df_sim['Residual Load'] - df_sim['Simulated Pumped Hydro'] - df_sim['Simulated Battery Storage']).max()
    if unmet_demand_peak > 0:
        unmet_demand_gw = unmet_demand_peak / 1000
        # Formats the GW value with 2 decimals and replaces the dot with a comma
        unmet_gw_str = f"{unmet_demand_gw:.2f}".replace(".", ",")
        unmet_mw_str = f"{unmet_demand_peak:,.0f}".replace(",", ".")
        st.error(f"⚠️ **Grid Deficit:** Peak unbalance detected! Shortage of **{unmet_gw_str} GW** ({unmet_mw_str} MW). Increase storage capacities or fossil/RE generation factors.")
    else:
        st.success("🎉 **Grid Balanced:** The specified absolute storage capacities are completely sufficient to buffer the residual load fluctuations during this period!")

# ==========================================
# 7. STREAMLIT UI (Tabs)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Infrastructure Dashboard",
    "🏭 Sectoral Balance", 
    "📊 Filterable Live Data", 
    "🎛️ Live Simulation", 
    "❄️/☀️ Seasonal Deviations"
])

# ----------------- TAB 1: INFRASTRUCTURE DASHBOARD -----------------
with tab1:
    st.subheader("📍 National Infrastructure & Energy Balance")
    
    all_carriers_raw = network.generators['carrier'].unique()
    display_to_internal = {}
    for c in all_carriers_raw:
        disp = translation.get(c, c)
        if disp not in display_to_internal: display_to_internal[disp] = []
        display_to_internal[disp].append(c)
        
    display_options = sorted(list(display_to_internal.keys()))
    selected_display = st.multiselect("⚡ Filter Energy Sources:", options=display_options, default=display_options)
    
    selected_internal = []
    for d in selected_display: selected_internal.extend(display_to_internal[d])
    filtered_gens = network.generators[network.generators['carrier'].isin(selected_internal)]

    total_ps = filtered_gens[filtered_gens.carrier == 'Pumpspeicher'].p_nom.sum() if 'Pumpspeicher' in filtered_gens.carrier.values else 0
    total_gen_no_ps = filtered_gens[filtered_gens.carrier != 'Pumpspeicher'].p_nom.sum()
    total_battery = network.storage_units.p_nom.sum()
    total_cons_hh = sum(network.loads_t.p_set[f'Load_HH_{node}'].max() for node in nodes_dict.keys() if f'Load_HH_{node}' in network.loads_t.p_set)
    total_cons_ind = sum(network.loads_t.p_set[f'Load_IND_{node}'].max() for node in nodes_dict.keys() if f'Load_IND_{node}' in network.loads_t.p_set)
    total_cons = total_cons_hh + total_cons_ind
    net_balance = (total_gen_no_ps + total_ps) - total_cons

    fig_dash = make_subplots(
        rows=2, cols=2, column_widths=[0.75, 0.25], row_heights=[0.7, 0.3], horizontal_spacing=0.05, vertical_spacing=0.08,
        specs=[[{'type': 'geo', 'rowspan': 2}, {'type': 'pie'}], [None, {'type': 'xy'}]],
        subplot_titles=("Infrastructure & Local Generation", "National Mix", "Energy Balance (MW)")
    )

    shown_legend_items = set()
    def add_to_fig(trace, row, col, name, group):
        is_first = name not in shown_legend_items
        if is_first: shown_legend_items.add(name)
        trace.update(name=name, legendgroup=group, legendgrouptitle_text=f"<b>{group}</b>" if is_first else None, showlegend=is_first)
        fig_dash.add_trace(trace, row=row, col=col)

    for i_line, line_data in network.lines.iterrows():
        b0_data, b1_data = network.buses.loc[line_data.bus0], network.buses.loc[line_data.bus1]
        add_to_fig(go.Scattergeo(lon=[b0_data.x, b1_data.x], lat=[b0_data.y, b1_data.y], mode='lines', line=dict(width=1, color=color_map.get("AC Grid", "gray")), opacity=0.3), 1, 1, "AC Grid", "Infrastructure")

    add_to_fig(go.Scattergeo(lon=[7.5, nodes_dict["Ulm"][0]], lat=[55.0, nodes_dict["Ulm"][1]], mode='lines', line=dict(width=3, color=color_map.get("SuedLink (DC)", "red"), dash='dash')), 1, 1, "SuedLink (DC)", "Infrastructure")

    offshore_gen = filtered_gens[filtered_gens.carrier.isin(['Windenergie (Offshore-Anlage)', 'Wind (offshore)'])].p_nom.sum()
    if offshore_gen > 0:
        add_to_fig(go.Scattergeo(lon=[7.5], lat=[55.0], mode='markers', marker=dict(size=18, color=color_map.get("Wind Offshore", "cyan"), symbol='star'), text=f"<b>Offshore Hub</b><br>Wind Offshore: {offshore_gen:,.0f} MW", hoverinfo='text'), 1, 1, "Wind Offshore", "Generation")

    radius_val = 24
    for bus_nm, (lon_val, lat_val) in nodes_dict.items():
        bus_gens = filtered_gens[filtered_gens.bus == bus_nm]
        mix_bus = bus_gens[~bus_gens.carrier.isin(['Windenergie (Offshore-Anlage)', 'Wind (offshore)'])].groupby('carrier').p_nom.sum().sort_values(ascending=False)
        total_bus = mix_bus.sum()

        hover_parts = [f"<b>BUS: {bus_nm}</b><br>"]
        if total_bus > 0:
            hover_parts.append("<b>Generation:</b>")
            for c, v in mix_bus.items(): 
                hover_parts.append(f"{translation.get(c,c)}: {v:,.0f} MW")
        try:
            h_load = network.loads_t.p_set[f'Load_HH_{bus_nm}'].max()
            i_load = network.loads_t.p_set[f'Load_IND_{bus_nm}'].max()
            hover_parts.append(f"<br><b>Demand:</b><br>HH: {h_load:,.0f} MW<br>Ind: {i_load:,.0f} MW")
        except: pass
        hover_text = "<br>".join(hover_parts)

        if total_bus > 0:
            cum_val = 0
            for carrier_nm, val_nm in mix_bus.items():
                trans_nm = translation.get(carrier_nm, carrier_nm)
                group_nm = "Storage" if trans_nm in ["Battery", "Pumped Hydro"] else "Generation"
                add_to_fig(go.Scattergeo(lon=[lon_val], lat=[lat_val], mode='markers',
                                        marker=dict(size=max(3, radius_val * (1 - cum_val)), color=color_map.get(trans_nm, 'gray')),
                                        hoverinfo='text', text=hover_text), 1, 1, trans_nm, group_nm)
                cum_val += val_nm/total_bus
        else:
            fig_dash.add_trace(go.Scattergeo(lon=[lon_val], lat=[lat_val], mode='markers', marker=dict(size=6, color='gray'), hoverinfo='text', text=hover_text, showlegend=False), row=1, col=1)
        fig_dash.add_trace(go.Scattergeo(lon=[lon_val], lat=[lat_val+0.25], mode='text', text=[bus_nm], textfont=dict(size=11, color='white'), showlegend=False), row=1, col=1)

    all_gen_summary = filtered_gens.groupby('carrier').p_nom.sum()
    if not all_gen_summary.empty:
        fig_dash.add_trace(go.Pie(labels=[translation.get(c,c) for c in all_gen_summary.index], values=all_gen_summary.values, marker=dict(colors=[color_map.get(translation.get(c,c), 'gray') for c in all_gen_summary.index]), hole=0.4, textinfo='percent', showlegend=False, domain={'x': [0.78, 1.0], 'y': [0.65, 0.95]}), row=1, col=2)

    labels_bar = ['Net Balance', 'Consumption', 'Battery Storage', 'Pumped Hydro', 'Generation']
    values_bar = [net_balance, -total_cons, total_battery, total_ps, total_gen_no_ps]
    colors_bar = ['#636EFA', '#EF553B', '#AA00FF', '#4B0082', '#00CC96']

    fig_dash.add_trace(go.Bar(x=values_bar, y=labels_bar, orientation='h', marker_color=colors_bar, text=[f"{v:,.0f} MW" for v in values_bar], textposition='outside', width=0.8, cliponaxis=False, showlegend=False), row=2, col=2)

    fig_dash.update_layout(template='plotly_dark', height=950, margin=dict(t=100, b=50, l=50, r=220), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig_dash.update_geos(
        scope='europe', lataxis_range=[47, 56], lonaxis_range=[5.5, 15.5], projection_type='mercator',
        showland=True, landcolor="#1f1f1f", showcountries=True, countrycolor="#555555", showocean=True, oceancolor="#0f0f0f"
    )
    fig_dash.update_yaxes(ticksuffix="   ", row=2, col=2)
    abs_mx = max(abs(min(values_bar)), max(values_bar)) * 1.6
    fig_dash.update_xaxes(range=[-abs_mx, abs_mx], row=2, col=2)
    
    st.plotly_chart(fig_dash, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Source: Bundesnetzagentur Deutschland")

# ----------------- TAB 2: SECTORAL BALANCE -----------------
with tab2:
    st.subheader("Sectoral Network Analysis")
    gen_capacity = network.generators.groupby("bus")['p_nom'].sum().reset_index()
    gen_capacity.columns = ['Node', 'Generation_Capacity_MW']

    data = []
    for bus in network.buses.index:
        try:
            peak_hh = network.loads_t.p_set[f"Load_HH_{bus}"].max()
            peak_ind = network.loads_t.p_set[f"Load_IND_{bus}"].max()
        except: peak_hh, peak_ind = 0, 0
        data.append({'Node': bus, 'Demand_Household_MW': peak_hh, 'Demand_Industry_MW': peak_ind})

    balance_df = pd.merge(pd.DataFrame(data), gen_capacity, on='Node', how='left').fillna(0)
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Bar(x=balance_df['Node'], y=balance_df['Demand_Industry_MW'], name='Max. Demand Industry', marker_color='#FFA15A'))
    fig_balance.add_trace(go.Bar(x=balance_df['Node'], y=balance_df['Demand_Household_MW'], name='Max. Demand Households', marker_color='#19D3F3'))
    fig_balance.add_trace(go.Bar(x=balance_df['Node'], y=balance_df['Generation_Capacity_MW'], name='Generation Capacity', marker_color='#00CC96'))
    fig_balance.update_layout(barmode='group', template='plotly_dark', yaxis_title="Megawatts (MW)", xaxis_title="Region (Node)")
    
    st.plotly_chart(fig_balance, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Source: Bundesnetzagentur Deutschland")

# ----------------- TAB 3: FILTERABLE LIVE DATA -----------------
with tab3:
    st.subheader("Analysis: Generation vs. ENTSO-E Actual Demand (Last 3 Days)")
    
    if not df_entsoe_live.empty and 'Actual Load' in df_entsoe_live.columns:
        x_axis = df_entsoe_live.index
        hours_index = x_axis.hour + x_axis.minute / 60
        sim_hh = (15000 + 10000 * np.sin(np.pi * (hours_index - 6) / 12) + 4000 * np.cos(np.pi * hours_index / 6)) * 2.5
        sim_ind = np.full(len(x_axis), 25000 * 1.8)
        sim_total = sim_hh + sim_ind
        
        df_entsoe_live['Demand: Households'] = df_entsoe_live['Actual Load'] * (sim_hh / sim_total)
        df_entsoe_live['Demand: Industry'] = df_entsoe_live['Actual Load'] * (sim_ind / sim_total)

        available_types = sorted([c for c in df_entsoe_live.columns if c not in ['Actual Load', 'Demand: Households', 'Demand: Industry']])
        selected_types = st.multiselect(
            "Select Data to Display (Generation & Demand):",
            options=available_types + ['Demand: Households', 'Demand: Industry', 'Actual Load'],
            default=available_types + ['Actual Load']
        )
        
        if selected_types:
            fig_entsoe = go.Figure()
            generation_types = [t for t in selected_types if t in available_types]
            for t in generation_types:
                fig_entsoe.add_trace(go.Scatter(x=x_axis, y=df_entsoe_live[t], mode='lines', name=t, stackgroup='one', line=dict(width=0.5)))

            if 'Demand: Households' in selected_types:
                fig_entsoe.add_trace(go.Scatter(x=x_axis, y=df_entsoe_live['Demand: Households'], mode='lines', name='Demand: Households (Scaled)', line=dict(color='#19D3F3', width=2, dash='dot')))
            if 'Demand: Industry' in selected_types:
                fig_entsoe.add_trace(go.Scatter(x=x_axis, y=df_entsoe_live['Demand: Industry'], mode='lines', name='Demand: Industry (Scaled)', line=dict(color='#FFA15A', width=2, dash='dash')))
            if 'Actual Load' in selected_types:
                fig_entsoe.add_trace(go.Scatter(x=x_axis, y=df_entsoe_live['Actual Load'], mode='lines', name='ENTSO-E TOTAL DEMAND', line=dict(color='white', width=4)))

            fig_entsoe.update_layout(template='plotly_dark', hovermode='x unified', xaxis_title="Time (UTC)", yaxis_title="Power (MW)", height=600)
            
            st.plotly_chart(fig_entsoe, width='stretch')
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Source: ENTSO-E")

# ----------------- TAB 4: LIVE SIMULATION -----------------
with tab4:
    render_simulation_tab(df_entsoe_live, tab_desc="Live Data, Last 3 Days", prefix_key="live")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Source: ENTSO-E")

# ----------------- TAB 5: SEASONAL DEVIATIONS -----------------
with tab5:
    season_choice = st.radio(
        "Choose the season for the simulation:", 
        ["❄️ Winter (Dunkelflaute / Low Wind & Solar, Jan 2024)", "☀️ Summer (High Solar, Jun 2024)"], 
        horizontal=True
    )
    
    if "Winter" in season_choice:
        render_simulation_tab(df_entsoe_winter, tab_desc="Winter Data, Jan 15-18 2024", prefix_key="winter")
    else:
        render_simulation_tab(df_entsoe_summer, tab_desc="Summer Data, Jun 15-18 2024", prefix_key="summer")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Source: ENTSO-E")

# ==========================================
# 8. FOOTER / SIGNATURE
# ==========================================
st.markdown("---")
st.markdown("**Project Work from Emilia Seidel**<br>University of Ulm<br>Institute for Energy Conversion and Storage", unsafe_allow_html=True)
