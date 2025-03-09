############# LIBRARIES  ################

import streamlit as st 
import joblib
import pandas as pd
import re
from transformers import pipeline
from langdetect import detect
import plotly.express as px
from PIL import Image
#from package import sentiment_predictor

# Model definition
#model = pipeline("zero-shot-classification", 
#                            model='joeddav/xlm-roberta-large-xnli')

# Save the model
#joblib.dump(model, './model/sentiment_detet.pkl')



############# GBLOBAL VARIABLES ################
flag_pattern = u"\U0001F1E0-\U0001F1FF"

title = "Multi Language 🌎 Sentiment Classification Application "
app_dsc = "Your App to predict the sentiment of a text in <i>English</i>, <i>French</i>, <i>German</i>, <i>Russian</i> or <i>Spanish</i>"
dags_hub_repo_info = "© By Zoumana Keita | Source Code on DagsHub."
metrics_info = "📊 Model Performances Per Language 👇🏼"
side_bar_info = "Change the metric in the sidebar (Top/Left)"
lang_error = "ℹ️ Please select one of the 5 languages above"
dags_hub_repo = "https://dagshub.com/zoumana.keita/Multi_language_Sentiment_Classification"
# Textual information

usage_options = """
    <div id="block1" style="float:left;">
        <h3>Option 1</h3>
        <ul>
            <li>Select a text to predict</li>
            <li>Click Predict</li>
        </ul>
    </div>

    <div id="block2" style="float:right;">
        <h3>Option 2</h3">
        <ul>
            <li>Provide Your own text</li>
            <li>Click Predict</li>
        </ul>
    </div>
"""
fr_text = "🇫🇷Je pense que tu peux mieux faire prochainement, n'abandonne surtout pas"
en_text = "🇺🇸This covid19 is becoming a nightmare. I hope things will get better soon"
ger_text = "🇩🇪ambiente eines billigen strandclubs in der türkei, der nachbar sitzt fast auf dem schoss weil es eng ist, die musik laut und unpassend ( fetenhits 80er ), gedränge und warme getränke die man gewöhnlich kalt trinkt. der eingang wird von 2 arroganten kleinen mädchen bedient, die sich auf irgendetwas was einbilden, unklar auf was. dazu gehen im laden afrikanische prostituierte auf männerfang. achja das essen: zu teuer, aber gut. für 1/3 des preises in anderen lokalen anzurufen. fazit: viel lärm um nichts"
rus_text = "🇷🇺спасибо большое отелю за хорошую организацию. в октябре арендовали зал для проведения тренинга для команды из 50 человек. все скромно, но достойно. менеджер андрей оперативно реагировал на все наши просьбы. например: произошла нестыковка кофе-брейка по времени, нужно было организовать перерыв раньше - через 10 мин все было накрыто. нужно было передвинуть столы - без проблем. кондиционер включить / выключить - сразу откликались. порадовало наличие бесплатной парковки. спасибо!"
sp_text = "🇪🇸Comida abundante, buena relacin calidad-precio si pides entrante + segundo se puede cenar por unos 20 euros"


############# HELPER FUNCTIONS  ################

@st.cache
def load_model():

    return joblib.load(open('./model/sentiment_detet.pkl','rb'))

    #model = pipeline("zero-shot-classification", 
    #                        model='joeddav/xlm-roberta-large-xnli')

    #return model

def load_metrics_data():

    return pd.read_csv("./data/metrics.csv")

@st.cache
def predict_sentiment(text, model):
    # Get all the candidate labels
    candidate_labels = ['Positive', 'Negative', 'Neutral']

    # Run the result
    result = model(text, candidate_labels, multi_class = True)

    del result["sequence"]
    result_df = pd.DataFrame(result)

    # Plot the result
    fig = px.bar(result_df, x='labels', y='scores')

    return fig

@st.cache
def show_metrics(result_df, metric):

    if(metric == "F1 Score"):
        fig = px.bar(result_df, x='languages', y='f1_scores', title="F1 Scores")

    elif(metric == "Accuracy"):
        fig = px.bar(result_df, x='languages', y='accuracy_scores', title="Accuracy Scores")

    return fig
    #st.plotly_chart(fig, use_container_width=True)


# Load the model
zsmlc_classifier = load_model()

# Read metrics data
metrics_df = load_metrics_data()

# Load the logo
image = Image.open('./images/dagshub_logo.png')
st.image(image)


st.markdown("<h1 style='text-align: center;'>"+title+"</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>"+app_dsc+"</h3>", unsafe_allow_html=True)

st.markdown("<a href= "+dags_hub_repo+"><p style= 'text-align: center;'>"+dags_hub_repo_info+"</p></a>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'> 💁🏽‍♂️ Usage of the App</h3>", unsafe_allow_html=True)

st.markdown(usage_options, unsafe_allow_html=True)


option = st.selectbox('',('', fr_text, en_text, ger_text, rus_text, sp_text))

with st.form(key='my_form'):
    message_container = st.empty()
    text_message = message_container.text_input("Your message")

    submit_button = st.form_submit_button('Predict')

    if(option):
        message_container.empty()
        option = re.sub(flag_pattern, '', option) # remove flags
        text_message = option

    if(submit_button):
        language_code = detect(text_message)
        if(language_code not in ['en', 'fr', 'de', 'ru', 'es']):
            st.markdown("<h4 style='color: red;'> "+lang_error+"</h4>", unsafe_allow_html=True)

        else:
            my_fig = predict_sentiment(text_message, zsmlc_classifier)
            st.plotly_chart(my_fig, use_container_width=True)

st.markdown("<h3 style='text-align: center;'>"+metrics_info+"</h3>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>"+side_bar_info+"</div>", unsafe_allow_html=True)

metric = st.sidebar.selectbox("Choice Model Performances 📊 To Plot", ["F1 Score", "Accuracy"])
metrics_fig = show_metrics(metrics_df, metric)
st.plotly_chart(metrics_fig, use_container_width=True)