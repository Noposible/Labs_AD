import numpy as np
from scipy.signal import butter, filtfilt
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output
import plotly.graph_objs as go

params = {
    'amplitude': 1.0,
    'frequency': 0.5,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_cov': 0.1,
    'cutoff': 5.0,
    'show_noise': ['show'],
    'graph_type': 'harmonic_noise',
    'filter_type': 'moving_average'
}

app = Dash(__name__)
server = app.server

N_POINTS = 500
t = np.linspace(0, 10, N_POINTS)
fs = N_POINTS / (t.max() - t.min())

slider_labels = {
    'amplitude': 'Амплітуда',
    'frequency': 'Частота',
    'phase': 'Фаза',
    'noise_mean': 'Середнє значення шуму',
    'noise_cov': 'Дисперсія шуму',
    'cutoff': 'Частота зрізу фільтра'
}

def generate_harmonic(amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def generate_noise(mean, covariance):
    std_dev = np.sqrt(max(covariance, 1e-9))
    return np.random.normal(mean, std_dev, size=t.shape)

def apply_filter(data, cutoff, method, fs=fs):
    nyq = 0.5 * fs
    cutoff = min(cutoff, nyq * 0.99)
    if cutoff <= 0 or len(data) < 15:
        return data
    if method == 'moving_average':
        b, a = butter(5, cutoff / nyq, btype='low')
        return filtfilt(b, a, data)
    elif method == 'exponential':
        dt = 1.0 / fs
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (rc + dt)
        filtered = np.empty_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]
        return filtered
    return data

def create_figure(amplitude, frequency, phase, noise_mean, noise_cov, cutoff, show_noise, graph_type, filter_type):
    harmonic = generate_harmonic(amplitude, frequency, phase)
    noise = generate_noise(noise_mean, noise_cov)
    filtered_noise = apply_filter(noise, cutoff, filter_type)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=harmonic, mode='lines', name='Гармоніка', line=dict(color='blue'), visible=True))
    fig.add_trace(
        go.Scatter(x=t, y=harmonic + noise, mode='lines', name='Гармоніка + Шум', line=dict(color='red', dash='dot'),
                   visible='show' in show_noise and graph_type == 'harmonic_noise'))
    fig.add_trace(go.Scatter(x=t, y=harmonic + filtered_noise, mode='lines', name='Гармоніка + Фільтрований шум',
                             line=dict(color='green', dash='dash'),
                             visible='show' in show_noise and graph_type == 'harmonic_filtered'))
    fig.update_layout(title='Графік гармоніки', xaxis_title='Час, с', yaxis_title='Амплітуда', uirevision='constant')
    return fig

sliders = [
    ('amplitude', 0.1, 2.0, None),
    ('frequency', 0.1, 2.0, None),
    ('phase', -np.pi, np.pi, None),
    ('noise_mean', -1.0, 1.0, None),
    ('noise_cov', 0.0, 1.0, None),
    ('cutoff', 0.1, max(50.0, fs / 2 - 1), None)
]

app.layout = html.Div([
    html.H2('Гармоніка з шумом та фільтрацією'),
    dcc.Dropdown(id='graph-type', options=[
        {'label': 'Чиста гармоніка', 'value': 'harmonic'},
        {'label': 'Гармоніка + шум', 'value': 'harmonic_noise'},
        {'label': 'Гармоніка + фільтрований шум', 'value': 'harmonic_filtered'}
    ], value=params['graph_type'], style={'width': '50%'}, clearable=False),
    dcc.Graph(id='harmonic-graph', figure=create_figure(**params)),
    html.Div([
        html.Div([
            html.Label(slider_labels[name]),
            dcc.Slider(id=name, min=min_, max=max_, step=step, value=params[name], updatemode='drag',
                       tooltip={"placement": "bottom", "always_visible": True})
        ]) for name, min_, max_, step in sliders
    ]),
    html.Label('Тип фільтра'),
    dcc.Dropdown(id='filter-type', options=[
        {'label': 'Рухоме середнє (Butterworth)', 'value': 'moving_average'},
        {'label': 'Експоненційний', 'value': 'exponential'}
    ], value=params['filter_type'], style={'width': '50%', 'marginBottom': '20px'}, clearable=False),
    dcc.Checklist(options=[{'label': 'Показати шум/фільтрований шум', 'value': 'show'}],
                  value=params['show_noise'], id='show-noise'),
    html.Button('Скинути', id='reset-button', n_clicks=0, style={'marginTop': '10px'}),
], style={'margin': '20px'})

@app.callback(
    Output('amplitude', 'value'), Output('frequency', 'value'), Output('phase', 'value'),
    Output('noise_mean', 'value'), Output('noise_cov', 'value'), Output('cutoff', 'value'),
    Output('show-noise', 'value'), Output('graph-type', 'value'), Output('filter-type', 'value'),
    Input('reset-button', 'n_clicks')
)
def reset(n_clicks):
    if n_clicks > 0:
        return (params['amplitude'], params['frequency'], params['phase'],
                params['noise_mean'], params['noise_cov'], params['cutoff'],
                params['show_noise'], params['graph_type'], params['filter_type'])
    return (no_update,) * 9

@app.callback(
    Output('harmonic-graph', 'figure'),
    Input('amplitude', 'value'), Input('frequency', 'value'), Input('phase', 'value'),
    Input('noise_mean', 'value'), Input('noise_cov', 'value'), Input('cutoff', 'value'),
    Input('show-noise', 'value'), Input('graph-type', 'value'), Input('filter-type', 'value'),
    prevent_initial_call=True
)
def update_graph(amplitude, frequency, phase, noise_mean, noise_cov, cutoff, show_noise, graph_type, filter_type):
    return create_figure(amplitude, frequency, phase, noise_mean, noise_cov, cutoff, show_noise, graph_type, filter_type)

if __name__ == '__main__':
    app.run(debug=True)
