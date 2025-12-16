import os
import re
import csv
import glob
import statistics

# Output files
RESULTS_DIR = "./profiling_results"
ROOFLINE_DAT = os.path.join(RESULTS_DIR, "roofline_data.dat")
TIME_DAT = os.path.join(RESULTS_DIR, "time_data.dat")
OCC_DAT = os.path.join(RESULTS_DIR, "occupancy_data.dat")
BENCH_FILE = os.path.join(RESULTS_DIR, "gpumembench.log")
SPECS_FILE = os.path.join(RESULTS_DIR, "roofline_specs.gp")

# Default GPU specs
SPECS = { 'bw_dram': 224.3, 'bw_l1': 28008.7, 'bw_shared': 2119.7, 'peak_flops': 155.7 }

def parse_gpumembench():
    if not os.path.exists(BENCH_FILE): return
    with open(BENCH_FILE, 'r') as f: content = f.read()
    dram = re.search(r'Global.*?read.*?:.*?([0-9\.]+)\s*GB/s', content, re.IGNORECASE)
    peak = re.search(r'Peak FP64.*:.*?([0-9\.]+)\s*GFLOP/s', content, re.IGNORECASE)
    if dram: SPECS['bw_dram'] = float(dram.group(1))
    if peak: SPECS['peak_flops'] = float(peak.group(1))

TRANS_SIZE = 32.0

def get_unit_multiplier(unit_str):
    if not unit_str: return 1.0
    u = unit_str.lower().strip()
    return 1e-3 if u == 'ms' else 1e-6 if u == 'us' else 1e-9 if u == 'ns' else 1.0

def read_csv_with_units(path):
    if not os.path.exists(path): return []
    with open(path, 'r', newline='') as f: lines = f.readlines()
    
    header_idx = -1
    for i, L in enumerate(lines[:20]):
        if '"Name"' in L or 'Name' in L: header_idx = i; break
    if header_idx == -1: return []

    keys = [k.strip().replace('"', '') for k in lines[header_idx].strip().split(',')]
    unit_map = {}
    data_start = header_idx + 1
    
    if len(lines) > header_idx + 1:
        units = [u.strip().replace('"', '') for u in lines[header_idx+1].strip().split(',')]
        if 's' in units or 'us' in units or 'ms' in units:
            data_start = header_idx + 2
            for i, u in enumerate(units):
                if i < len(keys): unit_map[keys[i]] = get_unit_multiplier(u)

    reader = csv.DictReader(lines[data_start:], fieldnames=keys)
    rows = []
    for r in reader:
        c_row = r.copy()
        for k, v in r.items():
            if k in unit_map and v:
                try: c_row[k] = float(re.sub(r'[^0-9\.]', '', v)) * unit_map[k]
                except: pass
        rows.append(c_row)
    return rows

def parse_elapsed_from_logs():
    times = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, '*.log')):
        name = os.path.basename(p).split('.')[0]
        with open(p, 'r') as f:
            content = f.read()
            # FIX: Match "Total Time: 14.22 s" OR standard nvprof "Elapsed time [s]:"
            m = re.search(r'(Total Time|Elapsed time \[s\])[:\s]+([0-9\.]+)', content)
            if m: times[name] = float(m.group(2))
    return times

def clean_version(base_name):
    s = base_name.replace('sciara_cuda_', '').replace('sciara_', '')
    s = re.sub(r'[_\-]+', ' ', s).strip().title()
    return 'Global' if s.lower() in ['cuda', 'global'] else s

def match_kernel_name(name):
    if not name: return None
    if 'computeOutflows' in name: return 'computeOutflows'
    if 'massBalance' in name: return 'massBalance'
    # FIX: Match both underscored and non-underscored versions
    if 'CfA' in name and 'Me' in name: return 'CfA_Me'
    if 'CfA' in name and 'Mo' in name: return 'CfA_Mo'
    return None

def parse_one_dataset(base):
    data = {'kernels': {}}
    base_path = os.path.join(RESULTS_DIR, base)
    
    # Read files
    sum_rows = read_csv_with_units(base_path + '_gpu_summary.csv')
    comp_rows = read_csv_with_units(base_path + '_compute.csv')
    mem_rows = read_csv_with_units(base_path + '_memory.csv')
    
    def get_k(k_name):
        if k_name not in data['kernels']:
            data['kernels'][k_name] = {'flops': 0.0, 'bytes': 0.0, 'time': 0.0}
        return data['kernels'][k_name]

    # Parse Time (Fixing overwrite bug)
    for r in sum_rows:
        k = match_kernel_name(r.get('Name', ''))
        if k and 'Time' in r:
            # FIX: Use += to sum multiple kernels (like CfAMo init+update)
            get_k(k)['time'] += float(r['Time'])

    # Parse Compute
    for r in comp_rows:
        k = match_kernel_name(r.get('Kernel', r.get('Name', '')))
        val = float(re.sub(r'[^0-9\.]', '', str(r.get('Metric Value', '0'))))
        if k and 'flop' in r.get('Metric Name', ''): get_k(k)['flops'] += val

    # Parse Memory
    for r in mem_rows:
        k = match_kernel_name(r.get('Kernel', r.get('Name', '')))
        val = float(re.sub(r'[^0-9\.]', '', str(r.get('Avg', '0'))))
        if k: get_k(k)['bytes'] += val * TRANS_SIZE

    return data

def main():
    parse_gpumembench()
    with open(SPECS_FILE, 'w') as f:
        for k, v in SPECS.items(): f.write(f"{k} = {v}\n")

    time_map = parse_elapsed_from_logs()
    roofline_rows = []

    for comp in glob.glob(os.path.join(RESULTS_DIR, '*_compute.csv')):
        base = os.path.basename(comp).replace('_compute.csv', '')
        version = clean_version(base)
        ds = parse_one_dataset(base)
        
        # Calculate Total Time if missing from log
        if base not in time_map:
            time_map[base] = sum(k['time'] for k in ds['kernels'].values())

        # Roofline Point
        total_flops = sum(k['flops'] for k in ds['kernels'].values())
        total_bytes = sum(k['bytes'] for k in ds['kernels'].values())
        # Use total application time for realistic GFLOPS
        total_time = time_map.get(base, 1.0) 

        if total_time > 0 and total_flops > 0:
            gflops = (total_flops / total_time) / 1e9
            ai = total_flops / max(1.0, total_bytes)
            roofline_rows.append({'label': version, 'ai': ai, 'gflops': gflops})

    # Write Outputs
    with open(ROOFLINE_DAT, 'w') as f:
        f.write('# Label AI GFLOPS Version\n')
        for r in roofline_rows:
            f.write(f'"{r["label"]}" {r["ai"]:.4f} {r["gflops"]:.4f} "{r["label"]}"\n')

    with open(TIME_DAT, 'w') as f:
        f.write('# Version Time_s\n')
        # Map filenames back to nice versions for the chart
        clean_times = {}
        for base, t in time_map.items():
            clean_times[clean_version(base)] = t
            
        for v, t in sorted(clean_times.items(), key=lambda x: x[1], reverse=True):
            f.write(f'"{v}" {t:.4f}\n')

    # Dummy occupancy (skipped for speed)
    with open(OCC_DAT, 'w') as f: f.write('# Version Occ\n')

    print("Data re-parsed successfully.")

if __name__ == '__main__':
    main()