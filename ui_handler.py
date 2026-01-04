import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.data_processor import AdvancedDataProcessor
from src.classifiers.factory import ClassifierFactory

class ComparisonDashboard(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Malware Tespiti: ID3 vs Gini vs Twoing Dashboard")
		try:
			self.state('zoomed')  # Windows için tam ekran
		except tk.TclError:
       # Eğer Windows değilse veya hata verirse:
            # Linux/Mac için manuel olarak ekran boyutunu alıp uygula
			w = self.winfo_screenwidth()
			h = self.winfo_screenheight()
			self.geometry(f"{w}x{h}")
		# Yollar ve Yöneticiler
		
		base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		dataset_path = os.path.join(base_dir, 'dataset', 'SOMLAP.csv')
	
		self.processor = AdvancedDataProcessor(dataset_path)
		self.factory = ClassifierFactory() 
	# Sonuçları saklamak için
		self.results = {} 
		self._init_ui()
	def _init_ui(self):
		main_frame = ttk.Frame(self)
		main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- ÜST KISIM: KONTROLLER ---
		controls_frame = ttk.LabelFrame(main_frame, text="Yöntem Kontrolleri")
		controls_frame.pack(fill="x", ipady=5)

		controls_frame.columnconfigure(0, weight=1)
		controls_frame.columnconfigure(1, weight=1)
		controls_frame.columnconfigure(2, weight=1)

		self.create_method_column(controls_frame, "ID3 (Entropy)", "id3", 0)
		self.create_method_column(controls_frame, "Gini (CART)", "gini", 1)
		self.create_method_column(controls_frame, "Twoing ", "twoing", 2)

        # --- ORTA KISIM: LOGLAR (YENİ TASARIM 4'LÜ EKRAN) ---
		log_frame = ttk.LabelFrame(main_frame, text="Log Merkezi")
		log_frame.pack(fill="x", pady=5, ipady=5)

        # Log widget'larını saklamak için sözlük
		self.log_widgets = {}

        # 4 Sütunlu Grid Yapısı: [Ana Sistem | ID3 | Gini | Twoing]
		log_configs = [
            ("main", "Ana Sistem", 0),
            ("id3", "ID3 Logları", 1),
            ("gini", "Gini Logları", 2),
            ("twoing", "Twoing Logları", 3)
        ]

		for key, title, col_idx in log_configs:
            # Her log için bir çerçeve
			frame = ttk.Frame(log_frame)
			frame.grid(row=0, column=col_idx, sticky="nsew", padx=2)
			log_frame.columnconfigure(col_idx, weight=1) # Eşit genişlik

            # Başlık
			lbl = ttk.Label(frame, text=title, font=("Arial", 8, "bold"), foreground="#555")
			lbl.pack(anchor="w")

            # Text Widget
			txt = tk.Text(frame, height=8, state='disabled', bg="#f8f9fa", font=("Consolas", 8), wrap=tk.WORD)
			txt.pack(fill="both", expand=True)
            
            # Referansı kaydet
			self.log_widgets[key] = txt

        # --- ALT KISIM: GRAFİKLER ---
		self.graph_frame = ttk.LabelFrame(main_frame, text="Dashboard & Karşılaştırma")
		self.graph_frame.pack(fill="both", expand=True)
	
	def create_method_column(self, parent, title, method_key, col_idx):
		frame = ttk.Frame(parent, borderwidth=2, relief="groove")
		frame.grid(row=0, column=col_idx, sticky="nsew", padx=5, pady=5)
		
		ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack(pady=5)
		
		# 1. Ön İşleme Butonu
		btn_prep = ttk.Button(frame, text="1. Ön İşleme Yap", 
							  command=lambda: self.run_preprocessing(method_key))
		btn_prep.pack(fill="x", padx=10, pady=2)
		
		# 2. Eğitim Butonu
		btn_train = ttk.Button(frame, text="2. Eğitimi Başlat", 
							   command=lambda: self.run_training(method_key))
		btn_train.pack(fill="x", padx=10, pady=2)
		
		# 3. Test Butonu
		btn_test = ttk.Button(frame, text="3. Test Et", 
							  command=lambda: self.run_testing(method_key))
		btn_test.pack(fill="x", padx=10, pady=2)
		
		# Durum Etiketi
		lbl_status = ttk.Label(frame, text="Durum: Bekleniyor", foreground="gray")
		lbl_status.pack(pady=5)
		
		# Referansı sakla (durumu güncellemek için)
		setattr(self, f"lbl_status_{method_key}", lbl_status)

	def log(self, msg, target="main"):

        # Eğer target sözlükte yoksa 'main' e yaz
		widget = self.log_widgets.get(target, self.log_widgets["main"])      
		widget.config(state='normal')
		widget.insert(tk.END, f">> {msg}\n")
		widget.see(tk.END) # Otomatik scroll
		widget.config(state='disabled')

	def update_status(self, method_key, text, color="blue"):
		lbl = getattr(self, f"lbl_status_{method_key}")
		lbl.config(text=f"Durum: {text}", foreground=color)

	# --- PROCESS TRIGGER FUNCTIONS ---

	def run_preprocessing(self, method):
		def task():
			try:
				self.update_status(method, "İşleniyor...", "orange")
                # Log hedefi belirtiyoruz: target=method
				self.log("Ön işleme başlatıldı...", target=method)
                
				if method == 'id3':
					tr, te = self.processor.process_for_id3()
				elif method == 'gini':
					tr, te = self.processor.process_for_gini()
				elif method == 'twoing':
					tr, te = self.processor.process_for_twoing()
                
				self.log(f"Tamamlandı. Train: {tr}, Test: {te}", target=method)
				self.log("Veriler kaydedildi.", target=method)
				self.update_status(method, "Veri Hazır", "green")
                
				self.log(f"{method.upper()} ön işlemesi bitti.", target="main")
                
			except Exception as e:
				self.log(f"HATA: {e}", target=method) # Hatayı ilgili kutuya yaz
				self.log(f"Sistem Hatası: {method} işleminde hata.", target="main") # Genel kutuya da yaz
				self.update_status(method, "Hata Oluştu", "red")
        
		threading.Thread(target=task, daemon=True).start()

	def run_training(self, method):
		def task():
			try:
				self.update_status(method, "Eğitiliyor...", "orange")
				self.log("Eğitim verisi yükleniyor...", target=method)
				X_train, y_train = self.processor.load_train_data(method)
                
				self.log("Model eğitiliyor...", target=method)
                
				classifier = self.factory.get_classifier(method)
				classifier.train(X_train, y_train)
                
				self.log("Model başarıyla eğitildi.", target=method)
				self.update_status(method, "Model Eğitildi", "green")
			except Exception as e:
				self.log(f"HATA: {e}", target=method)
				self.update_status(method, "Eğitim Hatası", "red")

		threading.Thread(target=task, daemon=True).start()

	def run_testing(self, method):
		def task():
			try:
				self.update_status(method, "Test Ediliyor...", "orange")
				self.log("Test verisi yükleniyor...", target=method)
				X_test, y_test = self.processor.load_test_data(method)
                
				metrics = self.factory.evaluate_model(method, X_test, y_test)
				self.results[method] = metrics
                
				res_str = f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f},Prec: {metrics['precision']:.4f}"
				self.log(f"Sonuçlar: {res_str}", target=method)
				self.update_status(method, "Test Bitti", "green")
                
				self.update_dashboard()
                
			except Exception as e:
				self.log(f"HATA: {e}", target=method)
				self.update_status(method, "Test Hatası", "red")

		threading.Thread(target=task, daemon=True).start()
	def update_dashboard(self):
		# Grafik alanını temizle
		for widget in self.graph_frame.winfo_children():
			widget.destroy()

		# Eğer hiç sonuç yoksa çık
		if not self.results:
			return

		# Figure Oluştur: 1 Satır, 2 Sütun (Sol: Bar Chart, Sağ: Confusion Matrices)
		# Confusion Matrix için sub-figure gerekebilir.
		fig = plt.Figure(figsize=(12, 5), dpi=100)
		
		# --- 1. Performans Karşılaştırma (Bar Chart) ---
		ax1 = fig.add_subplot(1, 2, 1)
		methods = list(self.results.keys())
		accuracies = [self.results[m]['accuracy'] for m in methods]
		f1_scores = [self.results[m]['f1'] for m in methods]
		precisions = [self.results[m]['precision'] for m in methods]
		
		x = range(len(methods))
		width = 0.25
		
		ax1.bar([i - width for i in x], accuracies, width, label='Accuracy', color='#4CAF50')
		ax1.bar([i for i in x], f1_scores, width, label='F1 Score', color='#2196F3')
		ax1.bar([i + width for i in x], precisions, width, label='Precision', color='#FF9800')
		
		ax1.set_ylabel('Skor')
		ax1.set_title('Modellerin Karşılaştırmalı Performansı')
		ax1.set_xticks(x)
		ax1.set_xticklabels([m.upper() for m in methods])
		ax1.set_ylim(0, 1.1)
		ax1.legend()
		ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

		# --- 2. Confusion Matrices (Truth Tables) ---
		gs = fig.add_gridspec(1, 2) # Ana grid zaten 1x2 idi.
		# Sağ tarafı 3'e bölelim (dikey)
		gs_right = gs[0, 1].subgridspec(3, 1)

		for idx, method in enumerate(methods):
			if idx > 2: break # Sadece 3 metoda yerimiz var
			ax_cm = fig.add_subplot(gs_right[idx])
			cm = self.results[method]['confusion_matrix']
			sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
			ax_cm.set_title(f'{method.upper()}', fontsize=9)
			ax_cm.set_ylabel('Gerçek')
			ax_cm.set_xlabel('Tahmin')

		fig.tight_layout()

		# Canvas'a çiz
		canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
		canvas.draw()
		canvas.get_tk_widget().pack(fill="both", expand=True)