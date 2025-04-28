#Завдання 1
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

"""
Інструкція для користувача:
- Слайдери змінюють параметри гармоніки та шуму.
- Чекбокс «Показати шум» вмикає/вимикає відображення зашумленого сигналу.
- Кнопка «Скинути» повертає всі параметри до початкових значень.
- Параметри шуму оновлюються миттєво без перегенерації гармоніки.
- Фільтр низьких частот згладжує шум із частотою зрізу, заданою відповідним слайдером.
"""
init_amplitude      = 1.0   # Початкова амплітуда
init_frequency      = 0.5   # Початкова частота
init_phase          = 0.0   # Початкова фаза (радіани)
init_noise_mean     = 0.0   # Початкове середнє шуму
init_noise_cov      = 0.1   # Початкова дисперсія шуму
init_cutoff         = 5.0   # Початкова частота зрізу фільтра
init_show_noise     = True  # Початково шум відображається

#Створення часового вектору
t = np.linspace(0, 10, 1000)  # від 0 до 10 секунд, 1000 точок

# Функція гармоніки
def generate_harmonic(amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

#Функція шуму
def generate_noise(mean, covariance):
    return np.random.normal(mean, np.sqrt(covariance), size=t.shape)

#Фільтр низьких частот
def lowpass_filter(data, cutoff, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

#Ініціалізація початкових сигналів
harmonic       = generate_harmonic(init_amplitude, init_frequency, init_phase)
noise          = generate_noise(init_noise_mean, init_noise_cov)
filtered_noise = lowpass_filter(noise, init_cutoff)
noisy_signal   = harmonic + filtered_noise

#Створення фігури та полів для віджетів
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, bottom=0.4)

#Малюємо початкові графіки
l_harmonic, = ax.plot(t, harmonic,      linestyle='--', color='navy',       label='Чиста гармоніка')
l_noisy,    = ax.plot(t, noisy_signal,  linestyle='-',  color='gold',        label='Гармоніка + шум')
l_filtered, = ax.plot(t, harmonic + filtered_noise,
                                linestyle='-',  color='forestgreen', label='Гармоніка + фільтрований шум')

ax.set_xlabel('Час, с')
ax.set_ylabel('Амплітуда')
ax.set_title('Гармоніка з шумом та фільтрацією')
ax.set_ylim(-2, 2)
ax.legend(loc='upper right')

#Поля для слайдерів
ax_amp    = plt.axes([0.2, 0.32, 0.65, 0.03])
ax_freq   = plt.axes([0.2, 0.28, 0.65, 0.03])
ax_phase  = plt.axes([0.2, 0.24, 0.65, 0.03])
ax_nmean  = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_ncov   = plt.axes([0.2, 0.16, 0.65, 0.03])
ax_cutoff = plt.axes([0.2, 0.12, 0.65, 0.03])

#Слайдери
s_amp    = Slider(ax_amp,    'Амплітуда',          0.1, 2.0,   valinit=init_amplitude)
s_freq   = Slider(ax_freq,   'Частота (Гц)',       0.1, 2.0,   valinit=init_frequency)
s_phase  = Slider(ax_phase,  'Фаза (рад)',        -np.pi, np.pi, valinit=init_phase)
s_nmean  = Slider(ax_nmean,  'Середнє шуму',      -1.0, 1.0,   valinit=init_noise_mean)
s_ncov   = Slider(ax_ncov,   'Дисперсія шуму',     0.001, 1.0, valinit=init_noise_cov)
s_cutoff = Slider(ax_cutoff, 'Частота зрізу (Гц)', 0.5, 50.0,  valinit=init_cutoff)

#Чекбокс для показу шуму
ax_check = plt.axes([0.8, 0.025, 0.15, 0.1])
check    = CheckButtons(ax_check, ['Показати шум'], [init_show_noise])

#Кнопка «Скинути»
reset_ax = plt.axes([0.1, 0.025, 0.1, 0.04])
button   = Button(reset_ax, 'Скинути')

#Функції оновлення
def update_harmonic(val):
    global harmonic
    harmonic = generate_harmonic(s_amp.val, s_freq.val, s_phase.val)
    l_harmonic.set_ydata(harmonic)
    if check.get_status()[0]:
        l_noisy.set_ydata(harmonic + noise)
    l_filtered.set_ydata(harmonic + filtered_noise)
    fig.canvas.draw_idle()

#Оновлює шум, фільтрує його і малює нові лінії
def update_noise(val):
    global noise, filtered_noise
    noise = generate_noise(s_nmean.val, s_ncov.val)
    filtered_noise = lowpass_filter(noise, s_cutoff.val)
    if check.get_status()[0]:
        l_noisy.set_ydata(harmonic + noise)
        l_noisy.set_visible(True)
    else:
        l_noisy.set_visible(False)
    l_filtered.set_ydata(harmonic + filtered_noise)
    fig.canvas.draw_idle()

#Увімкнення/вимкнення відображення шуму
def toggle_noise(label):
    visible = check.get_status()[0]
    l_noisy.set_visible(visible)
    fig.canvas.draw_idle()

#Скидає всі віджети до початкових значень
def reset(event):
    s_amp.reset();    s_freq.reset()
    s_phase.reset();  s_nmean.reset()
    s_ncov.reset();   s_cutoff.reset()

s_amp.on_changed(update_harmonic)
s_freq.on_changed(update_harmonic)
s_phase.on_changed(update_harmonic)
s_nmean.on_changed(update_noise)
s_ncov.on_changed(update_noise)
s_cutoff.on_changed(update_noise)
check.on_clicked(toggle_noise)
button.on_clicked(reset)
plt.show()
