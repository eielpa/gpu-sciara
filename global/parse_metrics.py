import os
import re
import csv
import glob
import statistics

# --- CONFIGURAZIONE ---
RESULTS_DIR = "./profiling_results"
ROOFLINE_DAT = os.path.join(RESULTS_DIR, "roofline_data.dat")
TIME_DAT = os.path.join(RESULTS_DIR, "time_data.dat")
OCC_DAT = os.path.join(RESULTS_DIR, "occupancy_data.dat")
BENCH_FILE = os.path.join(RESULTS_DIR, "gpumembench.log")
SPECS_FILE = os.path.join(RESULTS_DIR, "roofline_specs.gp")

# Fattori di scala
STEPS_PROFILE = 10.0
STEPS_FULL = 16000.0

# Default GPU specs (GTX 980)
SPECS = { 'bw_dram': 224.3, 'bw_l1': 28008.7, 'bw_shared': 2119.7, 'peak_flops': 155.7 }

def clean_key(k):
    """Rimuove virgolette e spazi dalle chiavi del CSV"""
    return k.strip().replace('"', '').replace("'", "")

def clean_val(v):
    """Rimuove caratteri non numerici dai valori"""
    if not v: return "0"
    return v.strip().replace('"', '').replace("'", "").replace("%", "")

def parse_csv_metrics(filepath):
    """Legge FLOPs e Bytes"""
    data = {'flops': 0.0, 'dram_bytes': 0.0}
    if not os.path.exists(filepath): return data
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    if len(lines) < 2: return data

    header_idx = -1
    for i, line in enumerate(lines[:20]):
        if "Metric Name" in line:
            header_idx = i
            break
    if header_idx == -1: return data

    keys = [clean_key(k) for k in lines[header_idx].strip().split(',')]
    
    try:
        idx_avg = -1
        idx_inv = -1
        idx_name = -1
        for i, k in enumerate(keys):
            if "Avg" in k: idx_avg = i
            if "Invocation" in k: idx_inv = i
            if "Metric" in k and "Name" in k: idx_name = i
    except: return data

    reader = csv.reader(lines[header_idx+1:])
    for row in reader:
        if not row or len(row) <= max(idx_avg, idx_inv, idx_name): continue
        try:
            metric = clean_val(row[idx_name])
            avg = float(clean_val(row[idx_avg]))
            inv = int(float(clean_val(row[idx_inv])))
            total_val = avg * inv
            
            if 'flop_count' in metric:
                data['flops'] += total_val
            elif 'dram_' in metric and '_transactions' in metric:
                data['dram_bytes'] += total_val * 32.0
        except: continue
    return data

def parse_occupancy_csv(filepath):
    """Legge l'occupazione media (Achieved Occupancy)"""
    values = []
    if not os.path.exists(filepath): return 0.0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    header_idx = -1
    for i, line in enumerate(lines[:20]):
        if "Metric Name" in line:
            header_idx = i
            break
    if header_idx == -1: return 0.0

    keys = [clean_key(k) for k in lines[header_idx].strip().split(',')]
    
    try:
        idx_avg = -1
        idx_name = -1
        for i, k in enumerate(keys):
            if "Avg" in k: idx_avg = i
            if "Metric" in k and "Name" in k: idx_name = i
    except: return 0.0

    reader = csv.reader(lines[header_idx+1:])
    for row in reader:
        if not row or len(row) <= max(idx_avg, idx_name): continue
        try:
            metric = clean_val(row[idx_name])
            if 'achieved_occupancy' in metric:
                val = float(clean_val(row[idx_avg]))
                # Se è percentuale (es. 45.2), converti in 0.452 se necessario, 
                # ma nvprof di solito dà il rapporto 0.0-1.0 o %.
                # Se il valore è > 1 assumiamo sia %, altrimenti ratio.
                # Qui facciamo media grezza.
                values.append(val)
        except: continue
        
    if not values: return 0.0
    return statistics.mean(values)

def parse_execution_time(log_path):
    if not os.path.exists(log_path): return 0.0
    with open(log_path, 'r') as f:
        content = f.read()
        m = re.search(r'(Total Time|Elapsed time \[s\])[:\s]+([0-9\.]+)', content)
        if m: return float(m.group(2))
    return 0.0

def parse_gpumembench():
    if not os.path.exists(BENCH_FILE): return
    with open(BENCH_FILE, 'r') as f: content = f.read()
    dram = re.search(r'Global.*?read.*?:.*?([0-9\.]+)\s*GB/s', content, re.IGNORECASE)
    peak = re.search(r'Peak FP64.*:.*?([0-9\.]+)\s*GFLOP/s', content, re.IGNORECASE)
    if dram: SPECS['bw_dram'] = float(dram.group(1))
    if peak: SPECS['peak_flops'] = float(peak.group(1))

def clean_version(base_name):
    s = base_name.replace('sciara_cuda_', '').replace('sciara_', '')
    s = re.sub(r'[_\-]+', ' ', s).strip().title()
    if s.lower() in ['cuda', 'global']: return 'Global'
    return s

def main():
    print("Parsing ALL metrics (Compute + Occupancy)...")
    parse_gpumembench()
    
    with open(SPECS_FILE, 'w') as f:
        for k, v in SPECS.items(): f.write(f"{k} = {v}\n")

    roofline_rows = []
    time_rows = []
    occ_rows = []

    files = glob.glob(os.path.join(RESULTS_DIR, '*_compute.csv'))
    
    for comp_file in files:
        base_name = os.path.basename(comp_file).replace('_compute.csv', '')
        version_label = clean_version(base_name)
        
        mem_file = comp_file.replace('_compute.csv', '_memory.csv')
        occ_file = comp_file.replace('_compute.csv', '_occupancy.csv') # File occupancy
        log_file = comp_file.replace('_compute.csv', '.log')
        
        # Parse Data
        compute_data = parse_csv_metrics(comp_file)
        mem_data = parse_csv_metrics(mem_file)
        occupancy_val = parse_occupancy_csv(occ_file) # Parse Occupancy
        total_time_full = parse_execution_time(log_file)
        
        # Roofline Calc
        flops_per_step = compute_data['flops'] / STEPS_PROFILE
        bytes_per_step = mem_data['dram_bytes'] / STEPS_PROFILE
        time_per_step = total_time_full / STEPS_FULL if total_time_full > 0 else 0.0

        if time_per_step > 0 and bytes_per_step > 0:
            gflops = (flops_per_step / time_per_step) / 1e9
            ai = flops_per_step / bytes_per_step
            roofline_rows.append((version_label, ai, gflops))
            
        time_rows.append((version_label, total_time_full))
        occ_rows.append((version_label, occupancy_val)) # Add to list

        print(f"[{version_label}] Occ: {occupancy_val:.4f}")

    # Write Outputs
    with open(ROOFLINE_DAT, 'w') as f:
        f.write('# Label AI GFLOPS Version\n')
        for label, ai, gflops in roofline_rows:
            f.write(f'"{label}" {ai:.6f} {gflops:.6f} "{label}"\n')

    with open(TIME_DAT, 'w') as f:
        f.write('# Version Time_s\n')
        for label, t in sorted(time_rows, key=lambda x: x[1], reverse=True):
            f.write(f'"{label}" {t:.4f}\n')

    # Write REAL Occupancy Data
    with open(OCC_DAT, 'w') as f:
        f.write('# Version Occupancy\n')
        for label, occ in sorted(occ_rows, key=lambda x: x[0]):
            f.write(f'"{label}" {occ:.6f}\n')

    print(f"\nDone! Occupancy data saved to {OCC_DAT}")

if __name__ == '__main__':
    main()