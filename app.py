from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import streamlit as st
from PIL import Image
import os
from mtcnn import MTCNN
import numpy as np
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))
def save_uploadded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False
def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    # Assuming 'face' is your input image array
    image = Image.fromarray(face)
    image = image.resize((200, 200))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('which famous celebrity are you?')

uploaded_image = st.file_uploader('choose an image')

if uploaded_image is not None:
    if save_uploadded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        st.image(display_image)

        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        st.text(features)
        st.text(features.shape)
        index_pos =recommend(feature_list,features)
        st.text(index_pos)
        col1, col2 = st.columns(2)
        with col1:
            st.header('your upoladed image')
            st.image(display_image)
        with col2:
            st.header(" ".join(filenames[index_pos].split('\\')[1].split('_')))
            st.image(filenames[index_pos],width=600)
