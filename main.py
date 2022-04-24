import os.path


import streamlit as st

import shutil

import voice_detect as vd

train_path = os.path.join("dataset","speeches")
# train_path_absolute ="C:\\Users\\Tanmay\\PycharmProjects\\Voice Detection\\dataset\\speeches"
json_path = "data.json"
model_path = "model.h5"
test_path = "test"
st.title('Voice Detection')

speaker_obj = vd.voice_detection_system()

# titles = vd.create_mapings("dataset/speeches")

# print(titles)

if st.checkbox("Test"):
    st.caption('changing radio button while training might cause unwanted issues')
    st.subheader('Testing...')



    uploaded_file = st.file_uploader("Choose a file", type=['wav'])
    if st.button("Test"):
        if uploaded_file is not None:
            st.markdown(
                """<h1 style='color:white;'>Audio : </h1>""",
                unsafe_allow_html=True)
            st.audio(uploaded_file)

            st.write("success")


           

            speaker, prediction, index = speaker_obj.predict(file_path =uploaded_file )
            if speaker == -1 and prediction == -1 and index == -1 :
                st.warning("signal length is short")
            # index = most_frequent(speaker_index)

            user = speaker_obj._mapping[index]
            print(user)

            st.markdown(
                """<h1 style='color:white;'>Prediction : </h1>""",
                unsafe_allow_html=True)

            st.write(f"The identify of speaker is {user} with probability of {prediction[0][index]} %")

            st.write(prediction)



