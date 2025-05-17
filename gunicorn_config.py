# gunicorn_config.py
bind = "0.0.0.0:10000"  # Puerto obligatorio en Render
workers = 1              # Solo 1 worker para ahorrar memoria
threads = 1              # 1 thread por worker
timeout = 120            # Tiempo máximo por request (segundos)
preload_app = True       # Carga la app antes de forking (crítico para ahorrar memoria)
max_requests = 100       # Reinicia worker cada 100 requests (previene leaks)
max_requests_jitter = 20 # Variación aleatoria para reinicios