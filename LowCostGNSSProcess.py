import os
import csv
import datetime
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Wedge, Circle
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
    """Mencari rnx2rtkp dari sistem Linux (Streamlit Cloud) atau folder lokal (Windows)."""
    # 1. Cek apakah rnx2rtkp terinstall di sistem Linux (Streamlit Cloud)
    system_path = shutil.which("rnx2rtkp")
    if system_path:
        return system_path

    # 2. Cek file lokal jika dijalankan di PC Windows lokal
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
    """Menhitung simpangan geodesi (dN, dE, dU) dalam meter dan azimuth vector."""
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
# 3. FUNGSIONALITAS STAGE 3: PLOTTING (LIMIT MAXIMUM 10 METER)
# =============================================================================
def get_quadrant_colors(u_arr, v_arr):
    colors_hex = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    colors = []
    for u, v in zip(u_arr, v_arr):
        if u >= 0 and v >= 0:
            colors.append(colors_hex[0])
        elif u < 0 and v >= 0:
            colors.append(colors_hex[1])
        elif u < 0 and v < 0:
            colors.append(colors_hex[2])
        else:
            colors.append(colors_hex[3])
    return colors

def generate_plots(df_vector, station_name, doy, start_hour):
    u_east = df_vector['dE_meter'].values
    v_north = df_vector['dN_meter'].values
    mag = df_vector['magnitude_2D_m'].values

    df_plot, obs_date_str, start_hr = prepare_datetime_with_utc_code(df_vector, doy, start_hour)

    fig = plt.figure(figsize=(18, 8.5), dpi=140)

    # SUBPLOT 1: TIME-SERIES QUIVER
    ax_ts = fig.add_subplot(1, 2, 1)
    ax_ts.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    y_base = np.zeros(len(df_plot))
    arrow_colors = get_quadrant_colors(u_east, v_north)

    ax_ts.quiver(
        df_plot['datetime'], y_base, 
        u_east, v_north, 
        angles='uv', scale_units='y', scale=1, 
        color=arrow_colors, width=0.0028, headwidth=3.8, 
        headlength=4.2, headaxislength=3.8, alpha=0.90, zorder=4
    )

    ax_ts.set_ylabel('Error Vector Magnitude & North Component (m)', fontsize=10, fontweight='bold')
    ax_ts.set_xlabel('Observation Time (UTC)', fontsize=10, fontweight='bold')
    
    # FORMAT JUDUL LAMA
    ax_ts.set_title(
        f'ERROR VECTOR TIME-SERIES PLOT\nSTATION: {station_name.upper()} | DATE: {obs_date_str} (Start: {start_hr:02d}:00 UTC)', 
        fontsize=11.5, fontweight='bold', pad=15
    )

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=30, ha='center')

    # BATAS SUMBU Y BATAS MAKSIMUM 10 METER
    ax_ts.set_ylim(-10.0, 10.0)
    ax_ts.grid(True, linestyle=':', alpha=0.6)

    # Inset Legenda Kuadran
    colors_quad = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    ax_inset = ax_ts.inset_axes([0.76, 0.62, 0.22, 0.33])
    ax_inset.set_aspect('equal')
    ax_inset.add_patch(Wedge((0, 0), 1, 0, 90, color=colors_quad[0], ec='white', lw=1.2))
    ax_inset.add_patch(Wedge((0, 0), 1, 90, 180, color=colors_quad[1], ec='white', lw=1.2))
    ax_inset.add_patch(Wedge((0, 0), 1, 180, 270, color=colors_quad[2], ec='white', lw=1.2))
    ax_inset.add_patch(Wedge((0, 0), 1, 270, 360, color=colors_quad[3], ec='white', lw=1.2))
    ax_inset.axhline(0, color='black', lw=0.8, zorder=5)
    ax_inset.axvline(0, color='black', lw=0.8, zorder=5)
    ax_inset.text(0.48, 0.48, 'Q-I\n(E-N)', ha='center', va='center', color='white', fontweight='bold', fontsize=5.5)
    ax_inset.text(-0.48, 0.48, 'Q-II\n(W-N)', ha='center', va='center', color='white', fontweight='bold', fontsize=5.5)
    ax_inset.text(-0.48, -0.48, 'Q-III\n(W-S)', ha='center', va='center', color='white', fontweight='bold', fontsize=5.5)
    ax_inset.text(0.48, -0.48, 'Q-IV\n(E-S)', ha='center', va='center', color='white', fontweight='bold', fontsize=5.5)
    ax_inset.add_patch(Circle((0, 0), 1, fill=False, edgecolor='black', lw=1.0))
    ax_inset.text(0, 1.25, 'N (+dN)', ha='center', va='bottom', fontsize=5.5, fontweight='bold')
    ax_inset.text(0, -1.25, 'S (-dN)', ha='center', va='top', fontsize=5.5, fontweight='bold')
    ax_inset.text(1.25, 0, 'E (+dE)', ha='left', va='center', fontsize=5.5, fontweight='bold')
    ax_inset.text(-1.25, 0, 'W (-dE)', ha='right', va='center', fontsize=5.5, fontweight='bold')
    ax_inset.set_xlim(-1.6, 1.6)
    ax_inset.set_ylim(-1.6, 1.6)
    ax_inset.axis('off')
    ax_inset.set_title('Quadrant Legend', fontsize=6.5, fontweight='bold', pad=3)

    # SUBPLOT 2: RADAR DISTRIBUTION (BATAS RADIUS MAKSIMUM 10 METER)
    ax_radar = fig.add_subplot(1, 2, 2, projection='polar')
    max_r = 10.0  # Ditetapkan tepat 10 meter
    
    ax_radar.set_theta_zero_location('N')
    ax_radar.set_theta_direction(-1)
    step = max_r / 5.0  # 2.0 meter per ring
    radii = np.arange(step, max_r + step, step)
    
    zone_colors = ['#f2f4f8', '#ffffff', '#e5e9f0', '#ffffff', '#d8dee9', '#ffffff']
    prev_r = 0
    theta_full = np.linspace(0, 2 * np.pi, 200)

    for idx, r in enumerate(radii):
        color = zone_colors[idx % len(zone_colors)]
        ax_radar.fill_between(theta_full, prev_r, r, color=color, alpha=0.65, zorder=1)
        prev_r = r

    ax_radar.set_rlim(0, max_r)
    ax_radar.set_rticks(radii)
    ax_radar.set_yticklabels([f"{r:.1f} m" for r in radii], fontsize=8, fontweight='bold', color='#2e3440')
    ax_radar.grid(True, linestyle='--', color='#4c566a', alpha=0.45, zorder=2)

    angles = np.radians([0, 45, 90, 135, 180, 225, 270, 315])
    labels = ['N (+dN)', 'NE', 'E (+dE)', 'SE', 'S (-dN)', 'SW', 'W (-dE)', 'NW']
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(labels, fontsize=8.5, fontweight='bold')

    theta_rad = np.arctan2(u_east, v_north)
    scatter = ax_radar.scatter(
        theta_rad, mag, c=mag, cmap='plasma',
        s=22, alpha=0.85, zorder=4, edgecolors='black', linewidths=0.3, vmin=0, vmax=max_r
    )

    cbar = fig.colorbar(scatter, ax=ax_radar, orientation='vertical', shrink=0.75, pad=0.1)
    cbar.set_label('Error Magnitude (m)', fontweight='bold', fontsize=9.5)

    # FORMAT JUDUL LAMA
    ax_radar.set_title(
        f'ERROR VECTOR DISTRIBUTION\nSTATION: {station_name.upper()} | DATE: {obs_date_str} | Total Epochs: {len(df_vector)}',
        fontsize=11.5, fontweight='bold', pad=15
    )

    plt.tight_layout()
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

                            # 4. PLOT VISUALISASI DENGAN SKALA 10 METER
                            st.subheader("📈 Vector & Radar Scatter Visualizations")
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
