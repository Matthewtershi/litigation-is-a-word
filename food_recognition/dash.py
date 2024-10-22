import streamlit as st
import tensorflow as tf
import numpy as np

# streamlit run dash.py
# possible python ux/uis include tkinter (figma translation(mid)), pyqt(best), streamlit(easy), dash(mostly for plots/plotly), Bokeh (complicated but great for data visualization)
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\matth\VSC\Projects\foodRecog\trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # returns index of max element

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])
if (app_mode == "Home"):
    st.header("Fruits and Vegetable Recognition System")
    # image_path = "home_img.jpg"
    # st.image(image_path)

elif (app_mode == "About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.markdown("""
    This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:

    **Fruits:**
    - Banana
    - Apple
    - Pear
    - Grapes
    - Orange
    - Kiwi
    - Watermelon
    - Pomegranate
    - Pineapple
    - Mango

    **Vegetables:**
    - Cucumber
    - Carrot
    - Capsicum
    - Onion
    - Potato
    - Lemon
    - Tomato
    - Radish
    - Beetroot
    - Cabbage
    - Lettuce
    - Spinach
    - Soybean
    - Cauliflower
    - Bell Pepper
    - Chilli Pepper
    - Turnip
    - Corn
    - Sweetcorn
    - Sweet Potato
    - Paprika
    - Jalapeño
    - Ginger
    - Garlic
    - Peas
    - Eggplant
    """)
    st.text("Pretty cool man")

elif (app_mode == "Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an image ...", type=['jpg', "jpeg", "png"])
    if (st.button("Show Image")):
        if (test_image is not None):
            st.image(test_image, width = 4, use_column_width=True)
        else:
            st.write("Please upload an image to display")
    
    if (st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting it's a {}".format(label[result_index]))


