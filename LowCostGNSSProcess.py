import os
import csv
import datetime
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import streamlit as st
import shutil

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Data Processing for Low Cost GNSS Network Indonesia",
    page_icon="📡",
    layout="wide"
)

# =============================================================================
# 1. FUNGSIONALITAS HELPER & STAGE 1: RTKLIB
# =============================================================================
def find_rnx2rtkp_executable():
    """Mencari rnx2rtkp dari sistem Linux (Streamlit Cloud) atau file lokal (Windows)."""
    system_path = shutil.which("rnx2rtkp")
    if system_path:
        return system_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "rnx2rtkp.exe"),
        os.path.join(script_dir, "rnx2rtkp"),
        os.path.join(script_dir, "rtklib_2.2.0", "rnx2rtkp.exe"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def find_reference_file():
    """Mencari file referensi koordinat di folder skrip."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ["FixedReferenceLowCostStation.txt", "FixedReferenceLowCostStation.csv"]:
        ref_path = os.path.join(script_dir, name)
        if os.path.exists(ref_path):
            return ref_path
    return None

def run_rtklib_processing(rnx2rtkp_path, obs_path, nav_path, output_pos_path):
    """Menjalankan rnx2rtkp dengan Single Point Positioning (-p 0) & mask 0 deg."""
    cmd = [
        rnx2rtkp_path,
        "-p", "0",
        "-m", "0",
        "-o", output_pos_path,
        obs_path,
        nav_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.exists(output_pos_path) and os.path.getsize(output_pos_path) > 0:
            return True, ""
        else:
            err_msg = result.stderr.strip() if result.stderr else "Output .pos tidak terbentuk atau kosong."
            return False, err_msg
    except Exception as e:
        return False, str(e)

def parse_pos_to_csv(pos_path, output_csv_path):
    """Mengekstrak data dari file .pos ke format CSV."""
    headers = [
        'Date', 'Time', 'Latitude', 'Longitude', 'Height', 
        'Q', 'ns', 'sdn', 'sde', 'sdu', 'sdne', 'sdeu', 'sdun', 'age', 'ratio'
    ]
    rows = []
    with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            parts = stripped.split()
            if len(parts) >= 5:
                rows.append(parts[:15])

    if rows:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return True, ""
    return False, "File .pos tidak berisi koordinat posisi valid."

# =============================================================================
# 2. FUNGSIONALITAS STAGE 2B: KALKULASI GEODESI
# =============================================================================
def parse_rinex_filename_info(filename):
    """Mengekstrak ID Stasiun (3 karakter), DOY, dan Kode Jam UTC dari nama file."""
    clean_name = os.path.basename(filename)
    for prefix in ['result_', 'hasil_', 'quiver_', 'deviation_']:
        if clean_name.lower().startswith(prefix):
            clean_name = clean_name[len(prefix):]
            
    clean_name = clean_name.split('.')[0]
    st_code = clean_name[:3].lower()
    
    doy = None
    start_hour = 0
    
    if len(clean_name) >= 8:
        doy_str = clean_name[4:7]
        if doy_str.isdigit():
            doy = int(doy_str)
            
        hour_char = clean_name[7].lower()
        if 'a' <= hour_char <= 'z':
            start_hour = ord(hour_char) - ord('a')
            
    return st_code, doy, start_hour, clean_name

def read_reference_file(ref_file_path):
    """Membaca file referensi (.txt / .csv)."""
    df_ref = pd.read_csv(ref_file_path, skipinitialspace=True)
    df_ref.columns = [col.strip() for col in df_ref.columns]

    required_cols = ['Station', 'ID', 'Lat', 'Lon', 'Alt']
    for col in required_cols:
        if col not in df_ref.columns:
            raise KeyError(f"Kolom '{col}' tidak ada dalam file referensi!")

    df_ref['stasiun_id'] = df_ref['ID'].astype(str).str.strip().str[:3].str.lower()
    return df_ref

def calculate_geodetic_displacements(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Menghitung simpangan geodesi (dN, dE, dU) dalam meter dan azimuth vector."""
    R_earth = 6378137.0  # WGS84 Radius (meter)
    lat_rad = np.radians(ref_lat)
    
    dLat_deg = lat - ref_lat
    dLon_deg = lon - ref_lon
    dAlt_m = alt - ref_alt
    
    dN_meter = np.radians(dLat_deg) * R_earth
    dE_meter = np.radians(dLon_deg) * R_earth * np.cos(lat_rad)
    dU_meter = dAlt_m
    
    mag_2D = np.sqrt(dE_meter**2 + dN_meter**2)
    mag_3D = np.sqrt(dE_meter**2 + dN_meter**2 + dU_meter**2)
    azimuth_deg = np.degrees(np.arctan2(dE_meter, dN_meter)) % 360.0
    
    return dLat_deg, dLon_deg, dAlt_m, dN_meter, dE_meter, dU_meter, mag_2D, mag_3D, azimuth_deg

def prepare_datetime_with_utc_code(df, doy, start_hour, year=2026):
    """Menyusun datetime UTC presisi berdasarkan Kode Jam dan DOY."""
    if doy is not None:
        base_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
        base_datetime = base_date.replace(hour=start_hour, minute=0, second=0)
    else:
        base_datetime = datetime.datetime(year, 1, 1, hour=start_hour)

    datetimes = []
    if 'Time' in df.columns:
        times = pd.to_numeric(df['Time'], errors='coerce').values
        if len(times) > 0 and not np.isnan(times[0]):
            first_sec = times[0]
            for t in times:
                dt = base_datetime + datetime.timedelta(seconds=float(t - first_sec))
                datetimes.append(dt)
        else:
            datetimes = [base_datetime + datetime.timedelta(seconds=i*30) for i in range(len(df))]
    else:
        datetimes = [base_datetime + datetime.timedelta(seconds=i*30) for i in range(len(df))]

    df['datetime'] = datetimes
    obs_date_str = base_datetime.strftime('%d %B %Y')
    return df, obs_date_str, start_hour

# =============================================================================
# 3. FUNGSIONALITAS STAGE 3: PROFESSIONAL LAYOUT PLOTTING
# =============================================================================
def generate_plots(df_vector, station_name, doy, start_hour):
    u_east = df_vector['dE_meter'].values
    v_north = df_vector['dN_meter'].values
    w_up = df_vector['dU_meter'].values
    mag = df_vector['magnitude_2D_m'].values

    df_plot, obs_date_str, start_hr = prepare_datetime_with_utc_code(df_vector, doy, start_hour)
    times = df_plot['datetime']

    # Set Theme & Figure Dimensions
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 9), dpi=150)
    
    # Header Utama Figure
    fig.suptitle(
        f"GNSS GEODETIC DISPLACEMENT & ERROR ANALYSIS\nSTATION: {station_name.upper()} | DATE: {obs_date_str} (Start: {start_hr:02d}:00 UTC)",
        fontsize=14, fontweight='bold', y=0.97, color='#1A202C'
    )

    # GridSpec: 3 Baris x 2 Kolom (Kiri: Time Series, Kanan: Radar & Panel Statistik)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1.0], hspace=0.28, wspace=0.25)

    # -------------------------------------------------------------------------
    # 1. TIME SERIES SUBPLOTS (SISI KIRI)
    # -------------------------------------------------------------------------
    ax_n = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0], sharex=ax_n)
    ax_u = fig.add_subplot(gs[2, 0], sharex=ax_n)

    # North-South Plot
    ax_n.plot(times, v_north, color='#E53E3E', linewidth=1.2, label='dN (North)')
    ax_n.axhline(0, color='#4A5568', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_n.set_ylabel('dNorth / NS (m)', fontsize=9.5, fontweight='bold', color='#2D3748')
    ax_n.set_ylim(-10.0, 10.0)
    ax_n.grid(True, linestyle=':', alpha=0.5)
    plt.setp(ax_n.get_xticklabels(), visible=False)

    # East-West Plot
    ax_e.plot(times, u_east, color='#3182CE', linewidth=1.2, label='dE (East)')
    ax_e.axhline(0, color='#4A5568', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_e.set_ylabel('dEast / EW (m)', fontsize=9.5, fontweight='bold', color='#2D3748')
    ax_e.set_ylim(-10.0, 10.0)
    ax_e.grid(True, linestyle=':', alpha=0.5)
    plt.setp(ax_e.get_xticklabels(), visible=False)

    # Altitude / Up Plot
    ax_u.plot(times, w_up, color='#38A169', linewidth=1.2, label='dU (Up)')
    ax_u.axhline(0, color='#4A5568', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_u.set_ylabel('dUp / Altitude (m)', fontsize=9.5, fontweight='bold', color='#2D3748')
    ax_u.set_xlabel('Observation Time (UTC)', fontsize=10, fontweight='bold', color='#2D3748')
    ax_u.set_ylim(-10.0, 10.0)
    ax_u.grid(True, linestyle=':', alpha=0.5)

    # Format Jam Sumbu X
    ax_u.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Legend Ringkas pada Subplot Kiri
    for ax in [ax_n, ax_e, ax_u]:
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------------------
    # 2. RADAR VECTOR DISTRIBUTION (SISI KANAN - ATAS)
    # -------------------------------------------------------------------------
    ax_radar = fig.add_subplot(gs[0:2, 1], projection='polar')
    max_r = 10.0
    
    ax_radar.set_theta_zero_location('N')
    ax_radar.set_theta_direction(-1)
    step = max_r / 5.0
    radii = np.arange(step, max_r + step, step)
    
    zone_colors = ['#EDF2F7', '#FFFFFF', '#E2E8F0', '#FFFFFF', '#CBD5E0', '#FFFFFF']
    prev_r = 0
    theta_full = np.linspace(0, 2 * np.pi, 200)

    for idx, r in enumerate(radii):
        color = zone_colors[idx % len(zone_colors)]
        ax_radar.fill_between(theta_full, prev_r, r, color=color, alpha=0.6, zorder=1)
        prev_r = r

    ax_radar.set_rlim(0, max_r)
    ax_radar.set_rticks(radii)
    ax_radar.set_yticklabels([f"{r:.1f}m" for r in radii], fontsize=7.5, color='#4A5568')
    ax_radar.grid(True, linestyle='--', color='#A0AEC0', alpha=0.5, zorder=2)

    angles = np.radians([0, 45, 90, 135, 180, 225, 270, 315])
    labels = ['N (+dN)', 'NE', 'E (+dE)', 'SE', 'S (-dN)', 'SW', 'W (-dE)', 'NW']
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(labels, fontsize=8.5, fontweight='bold', color='#2D3748')

    theta_rad = np.arctan2(u_east, v_north)
    scatter = ax_radar.scatter(
        theta_rad, mag, c=mag, cmap='plasma',
        s=22, alpha=0.85, zorder=4, edgecolors='black', linewidths=0.2, vmin=0, vmax=max_r
    )

    cbar = fig.colorbar(scatter, ax=ax_radar, orientation='vertical', shrink=0.75, pad=0.1)
    cbar.set_label('2D Error Mag (m)', fontweight='bold', fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax_radar.set_title('2D Vector Displacement Distribution', fontsize=10.5, fontweight='bold', pad=12, color='#1A202C')

    # -------------------------------------------------------------------------
    # 3. STATISTICAL SUMMARY PANEL (SISI KANAN - BAWAH)
    # -------------------------------------------------------------------------
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')

    # Kalkulasi Parameter Statistik Geodesi
    stats_data = [
        ["Component", "Min (m)", "Max (m)", "Mean (m)", "Std Dev / RMS (m)"],
        ["North (dN)", f"{np.min(v_north):.3f}", f"{np.max(v_north):.3f}", f"{np.mean(v_north):.3f}", f"{np.std(v_north):.3f}"],
        ["East (dE)", f"{np.min(u_east):.3f}", f"{np.max(u_east):.3f}", f"{np.mean(u_east):.3f}", f"{np.std(u_east):.3f}"],
        ["Up / Alt (dU)", f"{np.min(w_up):.3f}", f"{np.max(w_up):.3f}", f"{np.mean(w_up):.3f}", f"{np.std(w_up):.3f}"],
        ["2D Mag", f"{np.min(mag):.3f}", f"{np.max(mag):.3f}", f"{np.mean(mag):.3f}", f"{np.std(mag):.3f}"]
    ]

    table = ax_stats.table(
        cellText=stats_data,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.05, 0.96, 0.85]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Styling Tabel
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2B6CB0')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F7FAFC')
            else:
                cell.set_facecolor('#EDF2F7')
            cell.set_edgecolor('#CBD5E0')

    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.96)
    return fig

# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================
st.title("📡 Data Processing for Low Cost GNSS Network Indonesia")
st.markdown("Pengolahan data RINEX dari jaringan Low Cost GNSS Indonesia (Copyright: Kelompok Riset Ionosfer BRIN)")

# Sidebar
st.sidebar.header("⚙️ File Input & System Status")

rnx2rtkp_exe = find_rnx2rtkp_executable()
ref_path = find_reference_file()

if rnx2rtkp_exe:
    st.sidebar.success("✅ RTKLIB Engine Found")
else:
    st.sidebar.error("❌ 'rnx2rtkp.exe' Not Found!")

if ref_path:
    st.sidebar.success(f"✅ Reference File: {os.path.basename(ref_path)}")
else:
    st.sidebar.error("❌ 'FixedReferenceLowCostStation.txt' Not Found!")

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload Data RINEX")

obs_file = st.sidebar.file_uploader("1. Select Observation File (.o / .obs)", type=['o', 'obs', '26o', '24o', '23o'])
nav_file = st.sidebar.file_uploader("2. Select Navigation File (.p / .n / .nav)", type=['p', 'n', 'nav', '26p', '26n'])

# Button Processing
if st.sidebar.button("🚀 Process GNSS Data", type="primary"):
    if not rnx2rtkp_exe or not ref_path:
        st.error("Engine RTKLIB atau File Referensi belum lengkap di folder server.")
    elif not obs_file or not nav_file:
        st.warning("Silakan unggah kedua file (Observasi & Navigasi) terlebih dahulu.")
    else:
        with st.spinner("Processing GNSS Data (RTKLIB -> Stage 2B -> Vector Plots)..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Simpan file upload ke temp dir
                obs_temp_path = os.path.join(temp_dir, obs_file.name)
                nav_temp_path = os.path.join(temp_dir, nav_file.name)
                
                with open(obs_temp_path, "wb") as f:
                    f.write(obs_file.getbuffer())
                with open(nav_temp_path, "wb") as f:
                    f.write(nav_file.getbuffer())

                # 2. Eksekusi Stage 1 (RTKLIB)
                pos_output_path = os.path.join(temp_dir, "output.pos")
                csv_output_path = os.path.join(temp_dir, "output.csv")
                
                success, err_msg = run_rtklib_processing(rnx2rtkp_exe, obs_temp_path, nav_temp_path, pos_output_path)
                if not success:
                    st.error(f"RTKLIB Processing Failed: {err_msg}")
                else:
                    parse_success, debug_info = parse_pos_to_csv(pos_output_path, csv_output_path)
                    if not parse_success:
                        st.error(f"Parsing .pos to .csv failed: {debug_info}")
                    else:
                        # 3. Eksekusi Stage 2B
                        ref_df = read_reference_file(ref_path)
                        st_code, doy, start_hour, clean_name = parse_rinex_filename_info(obs_file.name)

                        match_ref = ref_df[ref_df['stasiun_id'] == st_code]
                        if match_ref.empty:
                            st.error(f"ID Stasiun '{st_code}' tidak ditemukan di file referensi!")
                        else:
                            ref_lat = float(match_ref.iloc[0]['Lat'])
                            ref_lon = float(match_ref.iloc[0]['Lon'])
                            ref_alt = float(match_ref.iloc[0]['Alt'])
                            station_name = str(match_ref.iloc[0]['Station'])

                            df_stage1 = pd.read_csv(csv_output_path)
                            dLat, dLon, dAlt, dN, dE, dU, mag2D, mag3D, azim = calculate_geodetic_displacements(
                                df_stage1['Latitude'].values, df_stage1['Longitude'].values, df_stage1['Height'].values,
                                ref_lat, ref_lon, ref_alt
                            )

                            df_simpangan = pd.DataFrame({
                                'date': df_stage1.get('Date', 'NODATE'),
                                'time': df_stage1.get('Time', 'NOTIME'),
                                'lat': df_stage1['Latitude'],
                                'lon': df_stage1['Longitude'],
                                'alt': df_stage1['Height'],
                                'dN_meter': dN,
                                'dE_meter': dE,
                                'dU_meter': dU,
                                'magnitude_2D_m': mag2D,
                                'magnitude_3D_m': mag3D,
                                'vector_azimuth_deg': azim
                            })

                            # RINGKASAN METRIK STASIUN
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Station Name", station_name)
                            col2.metric("Station ID", st_code.upper())
                            col3.metric("DOY / Start Hour", f"Day {doy} / {start_hour:02d}:00 UTC")
                            col4.metric("Total Epochs", len(df_simpangan))

                            st.markdown("---")

                            # 4. PLOT VISUALISASI BERTINGKAT
                            st.subheader("📈 Displacement Time-Series & Error Distribution Visualizations")
                            fig = generate_plots(df_simpangan, station_name, doy, start_hour)
                            st.pyplot(fig)

                            # 5. TABEL DATA & DOWNLOAD
                            st.markdown("---")
                            with st.expander("📊 View Data Table & Download Options"):
                                st.dataframe(df_simpangan, use_container_width=True)

                                csv_bytes = df_simpangan.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Full Vector Results (CSV)",
                                    data=csv_bytes,
                                    file_name=f"{clean_name}_vector.csv",
                                    mime="text/csv"
                                )
