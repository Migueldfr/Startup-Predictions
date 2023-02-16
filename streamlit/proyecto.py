from pyexpat import model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import seaborn as sns
import plotly.graph_objs as go
import streamlit.components.v1 as components
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import LabelEncoder

with open('streamlit/model_miguel_recall.pkl', 'rb') as re:
    recall_model = pickle.load(re)
    
with open('streamlit/model_miguel_precision.pkl', 'rb') as pre:
    precision_model = pickle.load(pre)

st.set_page_config(page_title="Startups", page_icon="money_with_wings:", layout="wide")

startups = "streamlit/startupready2.csv"
startups = pd.read_csv(startups, index_col='Unnamed: 0')

startups.rename(columns={"latidtud":"lat", "longitud":"lon"}, inplace=True)

img = Image.open('streamlit/introfoto.jpeg')

# Let,s do the feature engineering to get ready all features


startups['relation10'] = np.where(startups['relationships'].isin(range(11,64)),1,0)

startups['Has_roundABCD'] = np.where((startups['has_roundA'] ==1) & (startups['has_roundB'] == 1) & (startups['has_roundA'] == 1) & (startups['has_roundA'] == 1), 1,0)

startups['VCAN'] = np.where ((startups['has_VC'] == 1) & (startups['has_angel']),1,0)


startups['milesto_4'] = np.where(startups['milestones'].isin(range(1,4)),1,0)


startups['age-5'] = np.where(startups['age'].isin(np.arange(0,5,0.06)),1,0)
startups['age+10'] = np.where(startups['age'].isin(np.arange(12,15,0.01)),1,0)

startups['-5M'] = np.where(startups['funding_total_usd'].isin(np.arange(0,5*10**6)),1,0)
startups['+5/10M'] = np.where(startups['funding_total_usd'].isin(np.arange(11*10**6)),1,0)

# Let,s build a funtions for manual modeling
def classify(num):
    if num == 0:
        return 'Puede que nos juegue una mala pasada'
    if num == 1:
        return 'Tiene toda la pinta de ser exitosa'

def Modelo():
        st.title('Prediciones de los modelos')
        st.sidebar.header('Paramentros personalizados')
        # Funcion para poner los parametros del sidebar
        def user_input_parameters():
            relationships = st.sidebar.slider('Nº Relaciones', 0,64)
            age_last_milestone_year = st.sidebar.slider('Año de la empresa en el último hito', 0,50)
            milestones = st.sidebar.slider('Hitos', 0,50)
            Top500 = st.sidebar.selectbox('TOP 500?', ('Sí', 'No'), index=0)
            if Top500 == 'Sí':
                Top500 = 1
            else:
                Top500 = 0
            age = st.sidebar.slider('Años', 1,50)
            relation10 = st.sidebar.selectbox('Mas de 10 relaciones?', ('Sí', 'No'), index =0)
            if relation10 == 'Sí':
                relation10 = 1
            if relation10 =='No':
                relation10 = 0
            age_first_milestone_year = st.sidebar.slider('Año de la empresa en el primer hito', 0,10)
            Has_roundABCD = st.sidebar.selectbox('Tiene ronda A B C D', ('Sí', 'No'),index=0)
            if Has_roundABCD == 'Sí':
                Has_roundABCD = 1
            if Has_roundABCD =='No':
                Has_roundABCD = 0
            funding_rounds = st.sidebar.slider('Número de rondas', 0,10)
            has_roundB = st.sidebar.selectbox('Tiene la ronda B?', ('Sí', 'No'))
            if has_roundB == 'Sí':
                has_roundB = 1
            if has_roundB =='No':
                has_roundB = 0
            milesto_4 = st.sidebar.selectbox('Tiene menos de 4 hitos?', ('Sí', 'No'))
            if milesto_4 == 'Sí':
                milesto_4 = 1
            if milesto_4 =='No':
                milesto_4 = 0
            million = st.sidebar.selectbox('Alguno de sus hitos es relacionado con la palabra millones', ('Sí', 'No'))
            if million == 'Sí':
                million = 1
            if million =='No':
                million = 0
            data = {'relationships' : relationships,
                    'age_last_milestone_year':age_last_milestone_year,
                    'milestones':milestones,
                    'Top500':Top500,
                    'age':age,
                    'relation10':relation10,
                    'age_first_milestone_year':age_first_milestone_year,
                    'Has_roundABCD': Has_roundABCD,
                    'funding_rounds':funding_rounds,
                    'has_roundB':has_roundB,
                    'milesto_4':milesto_4,
                    'million':million}
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_parameters()


        # Escoger el modelo preferido
        option = ['RECALL', 'PRECISION']
        model = st.sidebar.selectbox('Que modelo quieres usar?', option)

        st.subheader('Parametros del cliente')
        st.subheader(model)
        st.write(df)

        if st.button('RUN'):
            if model == 'RECALL':
                st.success(classify(recall_model.predict(df)))
            elif model == 'PRECISION':
                st.success(classify(precision_model.predict(df)))
        
        st.write("[Canva](https://www.canva.com/design/DAFaqYdUSNU/5F0lbZqn6a3EHTPs-ceMnw/view?utm_content=DAFaqYdUSNU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)")


menu = st.sidebar.selectbox("Selecciona la página", ['Home','Modelo','Filtros'])

if menu == "Home":
    st.title("Predicciones Startups EEUU")

    st.image(img, use_column_width='auto')
    
    print('----------------------------------')


    st.title("Estado de las startups")
    
    st.subheader("Closed startups: " + str(startups['status'].value_counts()[0]))
    st.subheader("Operating startups: " + str(startups['status'].value_counts()[1]))

    status = startups['status'].value_counts(normalize=True)
    colors = sns.color_palette("Spectral").as_hex()


    fig = px.pie(values = status.values, names = ['Operating', 'Closed'], color_discrete_sequence=colors)
    st.plotly_chart(fig)

    st.markdown('Como podemos ver no esta desbanlanceado, mas adelante he probado a hacer undersamplig y oversamplin pero no hubo ningun cambio notable. Por lo que, por el momento se puede trabajar con el.')
    #st.markdown(" As we can see there is more Acquired companies than closed, but is affordable at the moment. It isn,t unbalanced, so we can work on it, I tried models with undersampling and oversampling but no changes happened. ")

    print('----------------------------------')

    st.title("Distribución de las startups por estados ")

    st.write(startups.head(5))
    st.map(startups)

    states = startups['state_code'].value_counts()[:10]
    fig_states = px.pie(values = states.values, names = states.index, title = 'Top 10 cantidad de startups')

    top10 = startups['category_code'].value_counts()[:10]
    fig_top10 = px.pie(values=top10, names = top10.index.map(str.capitalize), title="Top 10 mercados mas valorados $")

    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_states, use_container_width=True)
    right_column.plotly_chart(fig_top10, use_container_width=True)

    print('----------------------------------')

    st.title("Rango de relaciones empresariales")

    startups['range_relation'] = startups['relationships'].apply(lambda x: 'relationships 0' if x == 0 else 'relationships > 10' if x > 10 else 'relationships 1-10' )


    rate_success = startups.groupby(['range_relation','status']).agg({'iD':'count'}).reset_index()
    rate_success = pd.pivot_table(rate_success, values = 'iD',columns= ['status'], index= ['range_relation']).reset_index()
    rate_success.columns = ['Range','Closed','Operating']
    rate_success['Total'] = rate_success['Closed'] + rate_success['Operating'].astype(int)
    rate_success['Success Rate'] = round((rate_success['Operating'] / rate_success['Total'])*100,2).astype(float)
    rate_success = rate_success.sort_values(by= 'Success Rate')

    fig = px.bar(rate_success, x='Range', y=['Closed','Operating'])
    color = sns.set_palette("Spectral")
    fig.update_layout(barmode='group',bargroupgap=0.1)
    fig.update_layout(title_text='Distribution Success')
    fig.update_traces(marker=dict(color='lightsalmon'), selector=dict(name='Operating'))
    fig.update_traces(marker=dict(color='indianred'), selector=dict(name='Closed'))

    adquired_trace = fig.data[1]
    adquired_trace.update(text=rate_success['Success Rate'], texttemplate='%{text:.0f}%', textposition='outside')
    st.plotly_chart(fig,use_container_width=True)

    print('----------------------------------')

    st.title("Años en funcionamiento")

    startups['age']=startups['age'].astype(int)
    age = startups.groupby(['age','status']).agg({'iD':'count'}).reset_index()

    age = pd.pivot_table(age, values = 'iD', columns=['status'], index=['age']).reset_index()
    age.columns=['age','Closed','Acquired']

    fig = go.Figure()

    x_values = list(range(1, max(age['age'])+1))

    fig.add_trace(go.Bar(
        x=age['age'].values,
        y=age['Closed'].values,
        name='Closed',
        marker_color='indianred'))
    fig.add_trace(go.Bar(
        x=age['age'].values,
        y=age['Acquired'].values,
        name='Operating',
        marker_color='lightsalmon'
    ))

    adquired_trace = fig.data[0]
    adquired_trace.update(text=age['Closed'], texttemplate='%{text:.0f}', textposition='outside')

    adquired_trace = fig.data[1]
    adquired_trace.update(text=age['Acquired'], texttemplate='%{text:.0f}', textposition='outside')

    fig.update_layout(title_text='Years of operation',barmode='group',xaxis=dict(title = 'Años' ,tickmode='array', tickvals=x_values))
    st.plotly_chart(fig,use_container_width=True)

    print('----------------------------------')

    st.title("Total inversiones por estado/ciudad")

    founds = startups.groupby(['state_code','city'])[['funding_total_usd']].sum().sort_values('funding_total_usd',ascending=False).reset_index()
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]])

    fig.add_trace(go.Pie(labels = founds.state_code.values, values=founds.funding_total_usd.values, ),1,1)
    fig.add_trace(go.Pie(labels = founds.city.values, values=founds.funding_total_usd.values, ),1,2)

    fig.update_traces(textposition='inside', textinfo='percent+label',insidetextorientation='radial')


    fig.update_layout(title_text='Total funds ', width=1600,height=500, template = 'plotly_dark')
    st.plotly_chart(fig,use_container_width=True)

    print('----------------------------------')

    st.title("Rondas de inversión ")

    funding = startups.groupby(['funding_rounds','status'])[['status']].count().rename(columns={'status':'count'}).reset_index()
    funding = pd.pivot_table(funding, values = 'count',columns= ['status'], index= ['funding_rounds']).reset_index()
    funding.columns = ['funding_rounds','Closed','Acquired']

    fig = go.Figure()

    x_values = list(range(1, len(funding)+12))

    fig.add_trace(go.Bar(
        x=funding['funding_rounds'].values,
        y=funding['Closed'].values,
        name='Closed',
        marker_color='indianred'))
    fig.add_trace(go.Bar(
        x=funding['funding_rounds'].values,
        y=funding['Acquired'].values,
        name='Operating',
        marker_color='lightsalmon'
    ))

    adquired_trace = fig.data[0]
    adquired_trace.update(text=funding['Closed'], texttemplate='%{text:.0f}', textposition='outside')

    adquired_trace = fig.data[1]
    adquired_trace.update(text=funding['Acquired'], texttemplate='%{text:.0f}', textposition='outside')

    fig.update_layout(title_text='Funding rounds',xaxis=dict(title = 'Total funds' , tickmode='array', tickvals=x_values),barmode='group')

    st.plotly_chart(fig,use_container_width=True)

elif menu == "Filtros":
    st.sidebar.header('Opciones a filtrar: ')

    state = st.sidebar.multiselect("Seleccione el state: ",
                options = startups['state_code'].unique(),
                default= startups['state_code'].unique())
    
    istop = st.sidebar.multiselect("Es top 500: ",
                options = startups['Top500'].unique(),
                default= startups['Top500'].unique())
    
    milestones = st.sidebar.multiselect("Numero de hitos: ",
                options = startups['milestones'].unique(),
                default= startups['milestones'].unique())

    category = st.sidebar.multiselect("Categoria: ",
                options = startups['category_code'].unique(),
                default= startups['category_code'].unique())

    df_seleccion = startups.query('state_code == @state & Top500 == @istop & milestones == @milestones & category_code == @category')

    total_fundings = int(df_seleccion['funding_total_usd'].sum())

    is_top = int(df_seleccion['Top500'].count())

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader('Inversiones totales:')
        st.subheader(f"US $ {total_fundings:,}")

    with right_column:
        st.subheader('Número de Top 500 :')
        st.subheader(f" {is_top:}")

    st.markdown('---')

    st.dataframe(df_seleccion)

    relationships = (df_seleccion.groupby(['state_code']).mean()[['relationships']].sort_values(by = 'relationships'))

    fig_relation_cliente = px.bar(relationships, x = 'relationships', y = relationships.index, orientation = 'h', title = '<b>Relaciones de media por estado<b>', template = 'plotly_white',color='relationships', color_continuous_scale='Peach',
                           height=500, width=800)
    
    fig_relation_cliente.update_layout(plot_bgcolor = 'rgba(255,96,59)', xaxis = dict(showgrid = False,tickfont=dict(size=12)),
                                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                                coloraxis_colorbar=dict(title='Media de relaciones'),
                                bargap=0.3)

    st.plotly_chart(fig_relation_cliente,use_container_width=True)

    st.markdown('---')

    anio = (df_seleccion.groupby(['state_code']).mean()[['age']].sort_values(by = 'age'))

    fig_anio_cliente = px.bar(anio, x = 'age', y = anio.index, orientation = 'h', title = '<b>Años de media por estado<b>', template = 'plotly_white',color='age', color_continuous_scale='Oranges',
                           height=500, width=800)

    fig_anio_cliente.update_layout(plot_bgcolor = 'rgba(255,128,84)', xaxis = dict(showgrid = False, tickfont=dict(size=12)),
                                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                                coloraxis_colorbar=dict(title='Media de años'),
                                bargap=0.3)

    st.plotly_chart(fig_anio_cliente,use_container_width=True)

    st.markdown('---')

    miles = (df_seleccion.groupby(['state_code']).mean()[['milestones']].sort_values(by = 'milestones'))

    fig_miles_cliente = px.bar(miles, x = 'milestones', y = miles.index, orientation = 'h', title = '<b>Hitos de media por estado<b>', template = 'plotly_white',color='milestones', color_continuous_scale='OrRd',
                           height=500, width=800)

    fig_miles_cliente.update_layout(plot_bgcolor = 'rgba(255,158,110)', xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                                coloraxis_colorbar=dict(title='Media de hitos'),
                                bargap=0.3)

    st.plotly_chart(fig_miles_cliente,use_container_width=True)

    st.write("[Canva](https://www.canva.com/design/DAFaqYdUSNU/5F0lbZqn6a3EHTPs-ceMnw/view?utm_content=DAFaqYdUSNU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)")

elif menu == 'Modelo':
    Modelo()
    


hide_st_style = """ 
                <style>

                footer{visibility: hidden;}

                </style>
                """ 
st.markdown(hide_st_style, unsafe_allow_html=True)
