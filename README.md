# saw-karyawan
aplikasi pemilihan karyawan menggunakan metode SAW
#!/usr/bin/env python3
"""
SAW Pemilihan Karyawan - Modern Desktop App
Features:
- Modern UI (ttkbootstrap optional)
- Import CSV karyawan
- Setup kriteria (benefit/cost + bobot)
- Calculate SAW, show normalized matrix & ranking
- Bar chart (ranking) & Radar chart (profil karyawan)
- Export results CSV
- Save/Load project (.json): data + kriteria
"""

import csv, json, math, os, sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# optional libraries
try:
    import ttkbootstrap as tb
    TB = True
except Exception:
    TB = False

try:
    import numpy as np
except Exception:
    np = None

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None
    FigureCanvasTkAgg = None

try:
    from PIL import Image, ImageTk
    PIL = True
except Exception:
    PIL = False

APP_TITLE = "SAW - Pemilihan Karyawan Terbaik"

# ---------- Utilities ----------
def try_float(x):
    try:
        return float(x)
    except:
        return math.nan

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    headers = rows[0]
    data = rows[1:]
    return headers, data

def write_csv(path, headers, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# SAW computation (numpy if available)
def saw_rank(data_matrix, weights, types):
    if not data_matrix:
        return [], []
    if np is not None:
        A = np.array(data_matrix, dtype=float)
        m = A.shape[1]
        R = np.zeros_like(A)
        for j in range(m):
            col = A[:, j]
            if types[j] == 'benefit':
                maxv = np.nanmax(col)
                R[:, j] = 0 if (maxv == 0 or math.isnan(maxv)) else (col / maxv)
            else:  # cost
                minv = np.nanmin(col)
                with np.errstate(divide='ignore', invalid='ignore'):
                    R[:, j] = minv / col
                    R[~np.isfinite(R)] = 0
        W = np.array(weights, dtype=float)
        V = R.dot(W)
        return R.tolist(), V.tolist()
    else:
        # pure python fallback
        n = len(data_matrix); m = len(data_matrix[0])
        R = [[0.0]*m for _ in range(n)]
        for j in range(m):
            col = [data_matrix[i][j] for i in range(n)]
            if types[j] == 'benefit':
                maxv = max(col) if col else 0
                for i in range(n):
                    R[i][j] = (col[i] / maxv) if maxv not in (0, None) else 0.0
            else:
                minv = min(col) if col else 0
                for i in range(n):
                    try:
                        R[i][j] = (minv / col[i]) if col[i] != 0 else 0.0
                    except:
                        R[i][j] = 0.0
        V = []
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += R[i][j] * float(weights[j])
            V.append(s)
        return R, V

# ---------- App ----------
class SawKaryawanApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(1000, 640)

        # data holders
        self.karyawan = []  # list of {'name','values'}
        # default criteria example
        self.criteria = [
            {'name': 'Kompetensi', 'type': 'benefit', 'weight': 0.30},
            {'name': 'Pengalaman', 'type': 'benefit', 'weight': 0.25},
            {'name': 'Pendidikan', 'type': 'benefit', 'weight': 0.20},
            {'name': 'Kehadiran', 'type': 'benefit', 'weight': 0.15},
            {'name': 'Gaji', 'type': 'cost', 'weight': 0.10},
        ]
        self.last_R = []
        self.last_results = []

        self._build_ui()
        self.set_status("Ready")

    def set_status(self, text):
        try:
            self.status.config(text=text)
        except Exception:
            pass

    def _build_ui(self):
        # use tb.Window if available
        if TB:
            self.root = tb.Window(themename="flatly")
            self.root.title(APP_TITLE)

        # layout: sidebar + content
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=8)
        sidebar.grid(row=0, column=0, sticky="nsw")

        ttk.Label(sidebar, text="SAW Karyawan", font=("Segoe UI", 14, "bold")).pack(pady=(4,8))
        # nav buttons
        nav_btns = [
            ("Dashboard", self.show_dashboard),
            ("Data Karyawan", self.show_data),
            ("Kriteria", self.show_criteria),
            ("Perhitungan", self.show_calc),
            ("Hasil", self.show_result),
        ]
        for t, cmd in nav_btns:
            ttk.Button(sidebar, text=t, command=cmd).pack(fill='x', pady=4)

        ttk.Separator(sidebar).pack(fill='x', pady=8)
        ttk.Button(sidebar, text="Import CSV", command=self.import_csv).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Export Hasil", command=self.export_results).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Save Project", command=self.save_project).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Load Project", command=self.load_project).pack(fill='x', pady=4)
        if TB:
            ttk.Button(sidebar, text="Toggle Theme", command=self.toggle_theme).pack(fill='x', pady=6)

        # content frame (stacked pages)
        self.content = ttk.Frame(self.root, padding=10)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.pages = {}
        for name in ("dashboard","data","criteria","calc","result"):
            frm = ttk.Frame(self.content)
            frm.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = frm

        # build pages
        self._build_dashboard(self.pages['dashboard'])
        self._build_data(self.pages['data'])
        self._build_criteria(self.pages['criteria'])
        self._build_calc(self.pages['calc'])
        self._build_result(self.pages['result'])

        # status bar
        self.status = ttk.Label(self.root, text="Ready", anchor='w')
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.show_dashboard()

    # ---------- pages ----------
    def _make_card(self, parent, title, value):
        f = ttk.Frame(parent, relief='ridge', padding=10)
        ttk.Label(f, text=title).pack(anchor='w')
        lbl = ttk.Label(f, text=value, font=("Segoe UI", 20, "bold"))
        lbl.pack(anchor='center', pady=(6,0))
        f._value_label = lbl
        return f

    def _build_dashboard(self, frame):
        frame.columnconfigure((0,1,2), weight=1)
        self.card_total = self._make_card(frame, "Total Karyawan", "0")
        self.card_criteria = self._make_card(frame, "Total Kriteria", str(len(self.criteria)))
        self.card_top = self._make_card(frame, "Top Karyawan", "-")
        self.card_total.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.card_criteria.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        self.card_top.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")

        charts = ttk.Frame(frame)
        charts.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=8)
        charts.columnconfigure((0,1), weight=1)
        # bar chart holder
        bc = ttk.LabelFrame(charts, text="Ranking Chart")
        bc.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.bar_holder = ttk.Frame(bc); self.bar_holder.pack(fill='both', expand=True)
        # radar chart holder + selector
        rc = ttk.LabelFrame(charts, text="Radar Profil Karyawan")
        rc.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.radar_holder = ttk.Frame(rc); self.radar_holder.pack(fill='both', expand=True)
        sel = ttk.Frame(rc); sel.pack(fill='x')
        ttk.Label(sel, text="Pilih Karyawan:").pack(side='left', padx=6)
        self.radar_var = tk.StringVar()
        self.radar_combo = ttk.Combobox(sel, textvariable=self.radar_var, state='readonly')
        self.radar_combo.pack(side='left', padx=6)
        self.radar_combo.bind("<<ComboboxSelected>>", lambda e: self.update_radar())

    def _build_data(self, frame):
        top = ttk.Frame(frame)
        top.pack(fill='x', pady=6)
        ttk.Button(top, text="Import CSV", command=self.import_csv).pack(side='left', padx=6)
        ttk.Button(top, text="Tambah Karyawan", command=self.add_karyawan_popup).pack(side='left', padx=6)
        tbl = ttk.Frame(frame); tbl.pack(fill='both', expand=True)
        self.data_tree = ttk.Treeview(tbl, columns=("vals",), show='headings')
        self.data_tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(tbl, orient='vertical', command=self.data_tree.yview)
        sb.pack(side='right', fill='y'); self.data_tree.configure(yscrollcommand=sb.set)

    def _build_criteria(self, frame):
        top = ttk.Frame(frame); top.pack(fill='x', pady=6)
        ttk.Button(top, text="Tambah Kriteria", command=self.add_criteria_popup).pack(side='left', padx=6)
        ttk.Button(top, text="Auto Normalize Bobot", command=self.auto_normalize_weights_ui).pack(side='left', padx=6)
        tbl = ttk.Frame(frame); tbl.pack(fill='both', expand=True)
        self.crit_tree = ttk.Treeview(tbl, columns=("type","weight"), show='headings')
        self.crit_tree.heading("type", text="Tipe")
        self.crit_tree.heading("weight", text="Bobot")
        self.crit_tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(tbl, orient='vertical', command=self.crit_tree.yview)
        sb.pack(side='right', fill='y'); self.crit_tree.configure(yscrollcommand=sb.set)
        self.refresh_criteria_table()

    def _build_calc(self, frame):
        ttk.Label(frame, text="Tekan tombol untuk menghitung SAW berdasarkan data saat ini.").pack(anchor='w', pady=6)
        ttk.Button(frame, text="Hitung SAW", command=self.calculate_saw).pack(pady=6)
        self.norm_text = tk.Text(frame, height=12)
        self.norm_text.pack(fill='both', expand=True, pady=6)

    def _build_result(self, frame):
        top = ttk.Frame(frame); top.pack(fill='x', pady=6)
        ttk.Button(top, text="Export Hasil CSV", command=self.export_results).pack(side='left', padx=6)
        self.score_tree = ttk.Treeview(frame, columns=("score","rank"), show='headings')
        self.score_tree.heading("score", text="Score"); self.score_tree.heading("rank", text="Rank")
        self.score_tree.pack(fill='both', expand=True)

    # ---------- navigation ----------
    def show_dashboard(self): self._raise_page('dashboard'); self.update_dashboard()
    def show_data(self): self._raise_page('data'); self.refresh_data_table()
    def show_criteria(self): self._raise_page('criteria'); self.refresh_criteria_table()
    def show_calc(self): self._raise_page('calc')
    def show_result(self): self._raise_page('result'); self.refresh_result_table()

    def _raise_page(self, name):
        frame = self.pages[name]
        frame.tkraise()
        self.set_status(f"Viewing: {name}")

    # ---------- data functions ----------
    def import_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        headers, rows = read_csv(path)
        if len(headers) < 2:
            messagebox.showwarning("Format Error", "CSV harus berisi nama karyawan dan minimal satu kriteria.")
            return
        crit_names = headers[1:]
        self.karyawan = []
        for r in rows:
            name = r[0]
            vals = [try_float(x) for x in r[1:len(crit_names)+1]]
            self.karyawan.append({'name': name, 'values': vals})
        # if criteria length mismatch, build default from headers with equal weights
        if len(crit_names) != len(self.criteria):
            self.criteria = [{'name': cn, 'type':'benefit', 'weight': round(1.0/len(crit_names),3)} for cn in crit_names]
        self.refresh_all_after_data_change()
        self.set_status(f"Imported {len(self.karyawan)} karyawan, {len(self.criteria)} kriteria.")

    def add_karyawan_popup(self):
        win = tk.Toplevel(self.root); win.title("Tambah Karyawan")
        ttk.Label(win, text="Nama:").grid(row=0, column=0, padx=6, pady=6, sticky='w')
        name_var = tk.StringVar(); ttk.Entry(win, textvariable=name_var).grid(row=0, column=1, padx=6, pady=6)
        entries = []
        for i, c in enumerate(self.criteria):
            ttk.Label(win, text=f"{c['name']} ({c['type']}):").grid(row=1+i, column=0, sticky='w', padx=6, pady=4)
            v = tk.StringVar(); ttk.Entry(win, textvariable=v).grid(row=1+i, column=1, padx=6, pady=4)
            entries.append(v)
        def save():
            name = name_var.get().strip() or f"Karyawan {len(self.karyawan)+1}"
            vals = [try_float(e.get()) for e in entries]
            self.karyawan.append({'name': name, 'values': vals})
            self.refresh_all_after_data_change()
            win.destroy()
        ttk.Button(win, text="Simpan", command=save).grid(row=2+len(self.criteria), column=0, columnspan=2, pady=8)

    def refresh_data_table(self):
        cols = ["Karyawan"] + [c['name'] for c in self.criteria]
        self.data_tree["columns"] = cols
        for c in cols:
            self.data_tree.heading(c, text=c); self.data_tree.column(c, width=120, anchor='center')
        for r in self.data_tree.get_children(): self.data_tree.delete(r)
        for p in self.karyawan:
            vals = [p['name']] + [('' if math.isnan(v) else f"{v:g}") for v in p['values']]
            self.data_tree.insert("", "end", values=vals)
        self.radar_combo['values'] = [p['name'] for p in self.karyawan]

    # ---------- criteria functions ----------
    def refresh_criteria_table(self):
        for r in self.crit_tree.get_children(): self.crit_tree.delete(r)
        for c in self.criteria:
            self.crit_tree.insert("", "end", values=(c['type'], f"{c['weight']:g}"), text=c['name'])
        # show names by reconfiguring display: display text as first column (workaround)
        # (we keep it simple: types and weight shown, names visible via item text)
        # optionally one could rebuild columns to show names explicitly.

    def add_criteria_popup(self):
        win = tk.Toplevel(self.root); win.title("Tambah Kriteria")
        ttk.Label(win, text="Nama Kriteria:").grid(row=0, column=0, padx=6, pady=6, sticky='w')
        name_var = tk.StringVar(); ttk.Entry(win, textvariable=name_var).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(win, text="Tipe:").grid(row=1, column=0, padx=6, pady=6, sticky='w')
        type_var = tk.StringVar(value='benefit'); ttk.Combobox(win, values=['benefit','cost'], textvariable=type_var, state='readonly').grid(row=1, column=1, padx=6, pady=6)
        ttk.Label(win, text="Bobot (opsional):").grid(row=2, column=0, padx=6, pady=6, sticky='w')
        wvar = tk.StringVar(value='0'); ttk.Entry(win, textvariable=wvar).grid(row=2, column=1, padx=6, pady=6)
        def save():
            name = name_var.get().strip() or f"C{len(self.criteria)+1}"
            typ = type_var.get()
            w = try_float(wvar.get()); w = 0.0 if math.isnan(w) else float(w)
            self.criteria.append({'name': name, 'type': typ, 'weight': w})
            for p in self.karyawan:
                if len(p['values']) < len(self.criteria): p['values'] += [math.nan]
            self.refresh_criteria_table(); self.refresh_data_table()
            win.destroy()
        ttk.Button(win, text="Tambah", command=save).grid(row=3, column=0, columnspan=2, pady=8)

    def auto_normalize_weights_ui(self):
        # open a small popup that shows current weights and normalize
        total = sum([c['weight'] for c in self.criteria])
        if total == 0:
            n = len(self.criteria) or 1
            for c in self.criteria: c['weight'] = round(1.0/n,4)
        else:
            for c in self.criteria: c['weight'] = round(c['weight']/total,4)
        self.refresh_criteria_table()
        messagebox.showinfo("Normalized", "Bobot kriteria dinormalisasi (jumlah = 1).")

    # ---------- SAW calculation ----------
    def calculate_saw(self):
        if not self.karyawan:
            messagebox.showwarning("No Data", "Silakan import data karyawan terlebih dahulu.")
            return
        if not self.criteria:
            messagebox.showwarning("No Criteria", "Silakan tambahkan kriteria terlebih dahulu.")
            return
        mat = []
        for p in self.karyawan:
            row = []
            for v in p['values']: row.append(v if not math.isnan(v) else 0.0)
            mat.append(row)
        weights = [c['weight'] for c in self.criteria]
        types = [c['type'] for c in self.criteria]
        R, V = saw_rank(mat, weights, types)
        # show normalized
        self.norm_text.delete('1.0', tk.END)
        hdr = "\t".join([c['name'] for c in self.criteria]); self.norm_text.insert(tk.END, hdr + "\n")
        for i, r in enumerate(R[:200]):
            line = "\t".join([f"{x:.4f}" for x in r]); self.norm_text.insert(tk.END, f"{self.karyawan[i]['name']}\t{line}\n")
        # prepare results
        scored = [{'name': self.karyawan[i]['name'], 'score': V[i]} for i in range(len(V))]
        scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
        self.last_R = R; self.last_results = scored_sorted
        # update result table
        for r in self.score_tree.get_children(): self.score_tree.delete(r)
        for rank, itm in enumerate(scored_sorted, start=1): self.score_tree.insert("", "end", values=(f"{itm['score']:.6f}", rank), text=itm['name'])
        self.set_status("Perhitungan SAW selesai.")
        self.update_dashboard(); self.update_bar_chart()

    # ---------- charts ----------
    def update_bar_chart(self):
        for w in self.bar_holder.winfo_children(): w.destroy()
        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(self.bar_holder, text="Matplotlib tidak tersedia").pack()
            return
        if not getattr(self, 'last_results', None):
            ttk.Label(self.bar_holder, text="Belum ada hasil").pack()
            return
        labels = [it['name'] for it in self.last_results]
        values = [it['score'] for it in self.last_results]
        fig = Figure(figsize=(5,2.8), dpi=90); ax = fig.add_subplot(111)
        ax.bar(labels, values); ax.set_title("Ranking Karyawan (SAW)"); ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.bar_holder)
        canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_radar(self):
        for w in self.radar_holder.winfo_children(): w.destroy()
        if Figure is None or FigureCanvasTkAgg is None or np is None:
            ttk.Label(self.radar_holder, text="Numpy/Matplotlib diperlukan untuk radar").pack()
            return
        sel = self.radar_var.get()
        if not sel:
            ttk.Label(self.radar_holder, text="Pilih karyawan untuk melihat profil").pack()
            return
        prod = next((p for p in self.karyawan if p['name']==sel), None)
        if not prod:
            ttk.Label(self.radar_holder, text="Produk tidak ditemukan").pack()
            return
        labels = [c['name'] for c in self.criteria]
        data = [ (v if not math.isnan(v) else 0.0) for v in prod['values'] ]
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        vals = data + data[:1]; angs = angles + angles[:1]
        fig = Figure(figsize=(4,3.2), dpi=90); ax = fig.add_subplot(111, polar=True)
        ax.plot(angs, vals, marker='o'); ax.fill(angs, vals, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.radar_holder); canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)

    # ---------- dashboard ----------
    def update_dashboard(self):
        self.card_total._value_label.config(text=str(len(self.karyawan)))
        self.card_criteria._value_label.config(text=str(len(self.criteria)))
        top = self.last_results[0]['name'] if getattr(self, 'last_results', None) else "-"
        self.card_top._value_label.config(text=top)
        self.update_bar_chart()
        self.radar_combo['values'] = [p['name'] for p in self.karyawan]

    # ---------- results / export ----------
    def refresh_result_table(self):
        for r in self.score_tree.get_children(): self.score_tree.delete(r)
        if not getattr(self, 'last_results', None): return
        for rank, itm in enumerate(self.last_results, start=1): self.score_tree.insert("", "end", values=(f"{itm['score']:.6f}", rank), text=itm['name'])

    def export_results(self):
        if not getattr(self, 'last_results', None):
            messagebox.showinfo("No Results", "Belum ada hasil untuk di-export.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not out: return
        headers = ["Karyawan"] + [c['name'] for c in self.criteria] + ["Score", "Rank"]
        rows = []
        for rank, itm in enumerate(self.last_results, start=1):
            orig = next((p['values'] for p in self.karyawan if p['name']==itm['name']), [])
            rows.append([itm['name']] + [ ('' if math.isnan(v) else v) for v in orig ] + [itm['score'], rank])
        write_csv(out, headers, rows)
        self.set_status(f"Hasil diexport ke {out}"); messagebox.showinfo("Exported", f"Hasil diexport ke {out}")

    # ---------- save/load project ----------
    def save_project(self):
        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json")])
        if not out: return
        payload = {'karyawan': self.karyawan, 'criteria': self.criteria}
        try:
            with open(out,'w',encoding='utf-8') as f: json.dump(payload, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Project disimpan ke {out}")
            self.set_status(f"Project disimpan ke {out}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan project: {e}")

    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path: return
        try:
            with open(path,'r',encoding='utf-8') as f: payload = json.load(f)
            self.karyawan = payload.get('karyawan', []); self.criteria = payload.get('criteria', self.criteria)
            self.refresh_all_after_data_change()
            messagebox.showinfo("Loaded", f"Project dimuat dari {path}")
            self.set_status(f"Project dimuat dari {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat project: {e}")

    # ---------- helpers ----------
    def refresh_all_after_data_change(self):
        self.refresh_data_table(); self.refresh_criteria_table(); self.update_dashboard(); self.refresh_result_table()

    def refresh_data_table(self): self.refresh_data_table = lambda: None  # placeholder, replaced below
    # replace method with actual implementation to avoid forward-reference issue
    def refresh_data_table(self):
        cols = ["Karyawan"] + [c['name'] for c in self.criteria]
        self.data_tree["columns"] = cols
        for c in cols: self.data_tree.heading(c, text=c); self.data_tree.column(c, width=120, anchor='center')
        for r in self.data_tree.get_children(): self.data_tree.delete(r)
        for p in self.karyawan:
            vals = [p['name']] + [ ('' if math.isnan(v) else f"{v:g}") for v in p['values'] ]
            self.data_tree.insert("", "end", values=vals)
        self.radar_combo['values'] = [p['name'] for p in self.karyawan]

    def refresh_criteria_table(self):
        # overwritten earlier; keep function but rebuild tree
        try:
            for r in self.crit_tree.get_children(): self.crit_tree.delete(r)
            for c in self.criteria: self.crit_tree.insert("", "end", values=(c['type'], f"{c['weight']:g}"), text=c['name'])
        except Exception:
            pass

    # ---------- UI extras ----------
    def toggle_theme(self):
        if not TB:
            messagebox.showinfo("Info", "ttkbootstrap tidak terpasang.")
            return
        cur = self.root.style.theme_use()
        nxt = "darkly" if cur != "darkly" else "flatly"
        self.root.style.theme_use(nxt)

# ---------- run ----------
def main():
    if TB:
        root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()
    app = SawKaryawanApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
