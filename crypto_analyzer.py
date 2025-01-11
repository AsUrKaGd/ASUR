import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import logging
import threading
import time

# Настройка логирования
logging.basicConfig(filename='crypto_analyzer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация
CRYPTO_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
EXCHANGES = ['binance', 'exmo']
UPDATE_INTERVAL = 300  # Интервал обновления данных в секундах

# Функция для получения данных с Binance
def get_binance_data(symbol, interval='1h', limit=24):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Данные с Binance для {symbol} получены успешно.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при получении данных с Binance: {e}")
        return None

# Функция для получения данных с EXMO
def get_exmo_data(pair, limit=24):
    url = "https://api.exmo.com/v1.1/trades"
    params = {
        'pair': pair,
        'limit': limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Данные с EXMO для {pair} получены успешно.")
        return data[pair]
    except Exception as e:
        logging.error(f"Ошибка при получении данных с EXMO: {e}")
        return None

# Функция для обработки данных Binance
def process_binance_data(data):
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df

# Функция для обработки данных EXMO
def process_exmo_data(data):
    df = pd.DataFrame(data, columns=['trade_id', 'type', 'price', 'quantity', 'amount', 'date'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df['price'] = df['price'].astype(float)
    return df

# Функция для прогнозирования цены
def predict_price(df, hours=1):
    X = np.array(range(len(df))).reshape(-1, 1)  # Время как признак
    y = df['close'].values  # Цена как целевая переменная

    model = LinearRegression()
    model.fit(X, y)

    # Прогнозирование на заданное количество часов
    future_X = np.array([len(df) + i for i in range(hours)]).reshape(-1, 1)
    future_prices = model.predict(future_X)

    # Расчет вероятности (примерный расчет на основе R^2)
    r_squared = model.score(X, y)
    probability = round(r_squared * 100, 2)

    return future_prices, probability

# Функция для расчета RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для анализа данных
def analyze_data(symbol, exchange):
    if exchange == 'binance':
        data = get_binance_data(symbol)
        if data is None:
            return None
        df = process_binance_data(data)
    elif exchange == 'exmo':
        data = get_exmo_data(symbol)
        if data is None:
            return None
        df = process_exmo_data(data)
    else:
        logging.error("Неподдерживаемая биржа.")
        return None

    # Прогнозирование
    hours = int(hours_var.get())
    future_prices, probability = predict_price(df, hours)

    # Расчет RSI
    df['rsi'] = calculate_rsi(df)

    result = {
        'symbol': symbol,
        'exchange': exchange,
        'future_prices': future_prices.tolist(),
        'last_price': df['close'].iloc[-1],
        'rsi': df['rsi'].iloc[-1],
        'probability': probability
    }

    # Визуализация данных
    plot_prices(df, future_prices, symbol, exchange, hours)

    return result

# Функция для визуализации данных
def plot_prices(df, future_prices, symbol, exchange, hours):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], label='Исторические данные', marker='o')
    future_timestamps = [df['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(hours)]
    plt.plot(future_timestamps, future_prices, label='Прогноз', marker='x', linestyle='--')
    plt.xlabel('Время')
    plt.ylabel('Цена (USD)')
    plt.title(f'Цена {symbol} на {exchange.capitalize()}')
    plt.legend()
    plt.grid()
    plt.show()

# Функция для обработки нажатия кнопки "Анализировать"
def on_analyze():
    symbol = crypto_var.get()
    exchange = exchange_var.get()
    result = analyze_data(symbol, exchange)
    if result:
        result_label.config(text=f"Последняя цена: {result['last_price']:.2f} USD\n"
                                 f"Прогноз на {hours_var.get()} часов: {result['future_prices']}\n"
                                 f"Вероятность: {result['probability']}%\n"
                                 f"RSI: {result['rsi']:.2f}")
    else:
        messagebox.showerror("Ошибка", "Не удалось получить или обработать данные.")

# Функция для экспорта данных в CSV
def export_to_csv():
    symbol = crypto_var.get()
    exchange = exchange_var.get()
    if exchange == 'binance':
        data = get_binance_data(symbol)
        if data is None:
            return
        df = process_binance_data(data)
    elif exchange == 'exmo':
        data = get_exmo_data(symbol)
        if data is None:
            return
        df = process_exmo_data(data)
    else:
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Экспорт", "Данные успешно экспортированы в CSV.")

# Функция для автоматического обновления данных
def auto_update():
    while True:
        on_analyze()
        time.sleep(UPDATE_INTERVAL)

# Графический интерфейс
root = tk.Tk()
root.title("Анализатор крипторынка")
root.geometry("700x600")

# Вкладки
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Анализ')
tab_control.add(tab2, text='Настройки')
tab_control.pack(expand=1, fill='both')

# Элементы интерфейса на вкладке "Анализ"
label = ttk.Label(tab1, text="Выберите криптовалюту и биржу:")
label.pack(pady=10)

# Выпадающий список для выбора криптовалюты
crypto_var = tk.StringVar()
crypto_combobox = ttk.Combobox(tab1, textvariable=crypto_var)
crypto_combobox['values'] = CRYPTO_PAIRS
crypto_combobox.pack(pady=10)

# Выпадающий список для выбора биржи
exchange_var = tk.StringVar()
exchange_combobox = ttk.Combobox(tab1, textvariable=exchange_var)
exchange_combobox['values'] = EXCHANGES
exchange_combobox.pack(pady=10)

# Поле для ввода количества часов
hours_label = ttk.Label(tab1, text="Прогнозировать на (часов):")
hours_label.pack(pady=10)
hours_var = tk.StringVar(value="1")
hours_entry = ttk.Entry(tab1, textvariable=hours_var)
hours_entry.pack(pady=10)

# Кнопка для запуска анализа
analyze_button = ttk.Button(tab1, text="Анализировать", command=on_analyze)
analyze_button.pack(pady=20)

# Метка для вывода результата
result_label = ttk.Label(tab1, text="", font=('Arial', 12))
result_label.pack(pady=10)

# Кнопка для экспорта данных
export_button = ttk.Button(tab1, text="Экспорт в CSV", command=export_to_csv)
export_button.pack(pady=10)

# Запуск автоматического обновления в отдельном потоке
update_thread = threading.Thread(target=auto_update, daemon=True)
update_thread.start()

# Запуск основного цикла
root.mainloop()
