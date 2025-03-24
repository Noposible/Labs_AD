import streamlit as st
import pandas as pd
import plotly.express as px
import os

def load_vhi_data():
    folder = 'vhi'
    if not os.path.exists(folder):
        st.error("Папка 'vhi' не знайдена! Завантажте дані перед запуском.")
        return pd.DataFrame()

    files = [f for f in os.listdir(folder) if f.startswith('vhi_id_') and f.endswith('.csv')]
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    df_all = []

    area_names = {
        1: "Вінницька", 2: "Волинська", 3: "Дніпропетровська", 4: "Донецька", 5: "Житомирська",
        6: "Закарпатська", 7: "Запорізька", 8: "Івано-Франківська", 9: "Київська", 10: "Кіровоградська",
        11: "Луганська", 12: "Львівська", 13: "Миколаївська", 14: "Одеська", 15: "Полтавська",
        16: "Рівненська", 17: "Сумська", 18: "Тернопільська", 19: "Харківська", 20: "Херсонська",
        21: "Хмельницька", 22: "Черкаська", 23: "Чернівецька", 24: "Чернігівська", 25: "Крим"
    }

    for file_name in files:
        file_path = os.path.join(folder, file_name)
        df = pd.read_csv(file_path, header=1, names=headers, dtype=str, thousands=",")
        df = df.drop(columns=['empty'], errors='ignore')
        df['VHI'] = pd.to_numeric(df['VHI'], errors='coerce')
        df = df[df['VHI'] != -1].dropna()

        df['Year'] = df['Year'].str.extract(r'(\d{4})')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce', downcast="integer")
        df.dropna(subset=['Year'], inplace=True)
        df['Year'] = df['Year'].astype(int)

        area_id = int(file_name.split('_')[2])
        df['area'] = area_names.get(area_id, f"Область {area_id}")

        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)
    df_all.dropna(axis=1, how='all', inplace=True)

    return df_all

df = load_vhi_data()
if df.empty:
    st.stop()

df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

areas = sorted(df['area'].unique())
indices = ['VCI', 'TCI', 'VHI']

st.sidebar.header("Фільтри")
st.session_state.selected_index = st.sidebar.selectbox("Виберіть показник", indices, index=indices.index(st.session_state.get("selected_index", "VCI")))
st.session_state.selected_area = st.sidebar.selectbox("Виберіть область", areas, index=areas.index(st.session_state.get("selected_area", "Вінницька")))
st.session_state.week_range = st.sidebar.slider("Інтервал тижнів", int(df['Week'].min()), int(df['Week'].max()), st.session_state.get("week_range", (int(df['Week'].min()), int(df['Week'].max()))))
st.session_state.year_range = st.sidebar.slider("Інтервал років", int(df['Year'].min()), int(df['Year'].max()), st.session_state.get("year_range", (int(df['Year'].min()), int(df['Year'].max()))))
st.session_state.ascending = st.sidebar.checkbox("Сортувати за зростанням", value=st.session_state.get("ascending", False))
st.session_state.descending = st.sidebar.checkbox("Сортувати за спаданням", value=st.session_state.get("descending", False))

if st.session_state.ascending and st.session_state.descending:
    st.sidebar.warning("Оберіть лише один варіант сортування")
    st.session_state.ascending, st.session_state.descending = False, False

if st.sidebar.button("Скинути фільтри"):
    st.session_state.clear()
    st.rerun()

filtered_df = df[(df['area'] == st.session_state.selected_area) &
                 (df['Week'].between(*st.session_state.week_range)) &
                 (df['Year'].between(*st.session_state.year_range))]

if st.session_state.ascending:
    filtered_df = filtered_df.sort_values(by=st.session_state.selected_index, ascending=True)
elif st.session_state.descending:
    filtered_df = filtered_df.sort_values(by=st.session_state.selected_index, ascending=False)

col1, col2 = st.columns([1, 3])

with col2:
    tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік", "Порівняння областей"])

    with tab1:
        st.write("### Відфільтровані дані")
        st.dataframe(filtered_df.style.format({"Year": "{:.0f}"}))

    with tab2:
        st.write("### Графік змін показника")
        if filtered_df['Year'].nunique() == 1:
            fig = px.line(filtered_df, x='Week', y=st.session_state.selected_index,
                          title=f"Динаміка {st.session_state.selected_index} у {st.session_state.year_range[0]} році для області {st.session_state.selected_area}")
            fig.update_layout(xaxis_title="Тиждень")
        else:
            fig = px.line(filtered_df, x='Year', y=st.session_state.selected_index,
                          title=f"Динаміка {st.session_state.selected_index} для області {st.session_state.selected_area}")
        fig.update_layout(width=1000)
        st.plotly_chart(fig)

    with tab3:
        st.write("### Порівняння областей")
        comparison_df = df[(df['Week'].between(*st.session_state.week_range)) & (df['Year'].between(*st.session_state.year_range))]
        comparison_df[st.session_state.selected_index] = pd.to_numeric(comparison_df[st.session_state.selected_index], errors='coerce')
        comparison_df_aggregated = comparison_df.groupby(['Year', 'area'])[st.session_state.selected_index].mean().reset_index()
        filtered_comparison_df = comparison_df_aggregated[comparison_df_aggregated['Year'].between(*st.session_state.year_range)]
        fig_comp = px.line(filtered_comparison_df, x='Year', y=st.session_state.selected_index, color='area',
                           title=f"Порівняння {st.session_state.selected_index} по областях")
        fig_comp.update_traces(line=dict(width=1))
        fig_comp.update_traces(selector=dict(name=st.session_state.selected_area), line=dict(width=3, dash='solid'))
        fig_comp.update_layout(width=1000, height=800, title_x=0.5, xaxis_title="Рік", yaxis_title=st.session_state.selected_index,
                               template="plotly_dark", margin=dict(l=60, r=60, t=80, b=60))
        st.plotly_chart(fig_comp)