import streamlit as st
from PIL import Image
from ultralytics import YOLO

# load an official model
model = YOLO('yolov8n.pt')
# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

result_path = './content/results/predicted-melanoma-class.jpg'

# Show the results
def predict_img(img_path):
    # Run inference
    results = model(img_path)  # results list

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(result_path)  # save image

    return st.image(result_path)  # display image in Streamlit


DEMO_IMAGE = './content/demo_benign.jpg'

st.sidebar.write("**Use your own image**")
st.sidebar.write("**Your image is automatically converted to grayscale**")
img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if img_file_buffer is not None:
    UPLOADED_IMAGE = img_file_buffer
    UPLOADED_IMAGE = Image.open(UPLOADED_IMAGE).convert('L')
    DEMO_IMAGE = UPLOADED_IMAGE
    st.sidebar.image(DEMO_IMAGE, caption="grayScale")
    predict_img(UPLOADED_IMAGE)

    ## Save Result
    result_img = result_path
    with open(result_img, "rb") as file:
        btn = st.download_button(
            label="Save Result",
            data=file,
            file_name="melanoma-detection.jpg",
            mime="image/jpeg")
else:
    predict_img(DEMO_IMAGE)
