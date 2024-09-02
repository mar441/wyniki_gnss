import os
import pandas as pd
import matplotlib.pyplot as plt
import io
from flask import Flask
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np

def ecef_to_geodetic(x, y, z):
    a = 6378137.0  
    e = 8.1819190842622e-2  

    lon = np.arctan2(y, x)

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lat = np.arctan2(z + ep**2 * b * np.sin(th)**3, p - e**2 * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt

def load_pos_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_start_index = next(i for i, line in enumerate(lines) if line[0].isdigit())
    data = pd.read_csv(
        io.StringIO(''.join(lines[data_start_index:])),
        sep='\s+', 
        header=None,
        names=['GPST', 'x-ecef(m)', 'y-ecef(m)', 'z-ecef(m)', 'Q', 'ns', 'sdx(m)', 'sdy(m)', 'sdz(m)', 'sdxy(m)', 'sdyz(m)', 'sdzx(m)', 'age(s)', 'ratio']
    )
    
    data = data[['GPST', 'x-ecef(m)', 'y-ecef(m)', 'z-ecef(m)']]

    return data

def combine_pos_files(file_paths):
    combined_df = pd.concat([load_pos_file(fp) for fp in file_paths], ignore_index=True)
    return combined_df

receiver_files = {
    'Odbiornik_4': [os.path.join('odbiornik_1', 'pierwsza_godzina.pos'),
                    os.path.join('odbiornik_1', 'druga_godzina.pos'),
                    os.path.join('odbiornik_1', 'trzecia_godzina.pos'),
                    os.path.join('odbiornik_1', 'czwarta_godzina.pos')],
    'Odbiornik_3': [os.path.join('odbiornik_2', 'PIERWSZA_GODZINA.pos'),
                    os.path.join('odbiornik_2', 'DRUGA_GODZINA.pos'),
                    os.path.join('odbiornik_2', 'TRZECIA_GODZINA.pos'),
                    os.path.join('odbiornik_2', 'CZWARTA_GODZINA.pos')],
    'Odbiornik_2': [os.path.join('odbiornik_3', 'PIERWSZA_GODZINA.pos'),
                    os.path.join('odbiornik_3', 'DRUGA_GODZINA.pos'),
                    os.path.join('odbiornik_3', 'TRZECIA_GODZINA.pos'),
                    os.path.join('odbiornik_3', 'CZWARTA_GODZINA.pos')],
    'Odbiornik_1': [os.path.join('odbiornik_4', 'PIERWSZA_GODZINA.pos'),
                    os.path.join('odbiornik_4', 'DRUGA_GODZINA.pos'),
                    os.path.join('odbiornik_4', 'TRZECIA_GODZINA.pos'),
                    os.path.join('odbiornik_4', 'CZWARTA_GODZINA.pos')]
}


all_data = pd.DataFrame()
for receiver, files in receiver_files.items():
    combined_data = combine_pos_files(files)
    combined_data['receiver'] = receiver
    all_data = pd.concat([all_data, combined_data], ignore_index=True)


all_data['latitude'], all_data['longitude'], _ = zip(*all_data.apply(lambda row: ecef_to_geodetic(row['x-ecef(m)'], row['y-ecef(m)'], row['z-ecef(m)']), axis=1))

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    dcc.Graph(id='map'),
    html.Div(id='displacement-container', children=[
        dcc.Graph(id='displacement-graph-x'),
        dcc.Graph(id='displacement-graph-y'),
        dcc.Graph(id='displacement-graph-z')
    ], style={'display': 'none'}) 
])

@app.callback(
    Output('map', 'figure'),
    Input('map', 'id')
)
def update_map(_):
    fig = px.scatter_mapbox(
        all_data.drop_duplicates(subset=['receiver']), 
        lat='latitude', 
        lon='longitude', 
        hover_name='receiver', 
        color='receiver',
        zoom=5,  
        height=600  
    )
    min_lat = all_data['latitude'].min()
    max_lat = all_data['latitude'].max()
    min_lon = all_data['longitude'].min()
    max_lon = all_data['longitude'].max()

    fig.update_layout(mapbox_style="open-street-map", 
                      mapbox_center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2},
                      mapbox_zoom=5)

    fig.update_layout(mapbox_bounds={"west": min_lon - 0.0005, "east": max_lon + 0.0005, "south": min_lat - 0.0005, 
                                     "north": max_lat + 0.0005})
    
    fig.update_layout(legend_title_text='Odbiornik')
    return fig

@app.callback(
    [Output('displacement-graph-x', 'figure'),
     Output('displacement-graph-y', 'figure'),
     Output('displacement-container', 'style'),
     Output('displacement-graph-z', 'figure')],
    Input('map', 'clickData')
)
def display_displacement(clickData):
    if clickData is None:
        return {}, {}, {'display': 'none'}, {}
    
    receiver_name = clickData['points'][0]['hovertext']
    filtered_data = all_data[all_data['receiver'] == receiver_name].copy()

    fig_x = px.line(filtered_data, x='GPST', y='x-ecef(m)',  
                    title=f"Współrzędna X w czasie dla {receiver_name}",
                    markers=True)
    fig_x.update_layout(xaxis_title='Czas', yaxis_title='X [m]')
    fig_x.update_xaxes(nticks=10)
    fig_x.update_yaxes(tickformat=".4f") 

    fig_y = px.line(filtered_data, x='GPST', y='y-ecef(m)',  
                    title=f"Współrzędna Y w czasie dla {receiver_name}",
                    markers=True)
    fig_y.update_layout(xaxis_title='Czas', yaxis_title='Y [m]')
    fig_y.update_xaxes(nticks=10)
    fig_y.update_yaxes(tickformat=".4f") 

    fig_z = px.line(filtered_data, x='GPST', y='z-ecef(m)',  
                    title=f"Współrzędna Z w czasie dla {receiver_name}",
                    markers=True)
    fig_z.update_layout(xaxis_title='Czas', yaxis_title='Z [m]')
    fig_z.update_xaxes(nticks=10) 
    fig_z.update_yaxes(tickformat=".4f")  
    
    return fig_x, fig_y, {'display': 'block'}, fig_z

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
