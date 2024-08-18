import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def app():
  st.title("ねこかもしれない")
  model = YOLO('yolov8n.pt')
  class_names_map = model.names

  with st.form("my_form"):
    uploaded_image = st.file_uploader("ねこの画像を選択してください", type=["jpg", "jpeg", "png"])
    st.form_submit_button(label='猫を検知する')

  if uploaded_image:
        image = Image.open(uploaded_image)
        keys = [key for key, value in class_names_map.items() if value in "cat"]
        results = model(image, classes=keys)

        annotated_image = results[0].plot()  # OpenCV形式（BGR）で返される
        # StreamlitはPIL形式（RGB）を期待しているため BGRからRGBに変換
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_image_pil = Image.fromarray(annotated_image_rgb)
        if results[0].boxes.conf.numel() > 0:
          item_value = results[0].boxes.conf.item()
          if item_value > 0.8:
              st.markdown('## 猫です！！！')
              st.image(annotated_image_pil, caption="Detected Objects.", use_column_width=True)
          elif item_value > 0.6:
              st.markdown('## 多分猫です！')
              st.image(annotated_image_pil, caption="Detected Objects.", use_column_width=True)
          elif item_value > 0.4:
              st.markdown('## ねこかもしれない？')
              st.image(annotated_image_pil, caption="Detected Objects.", use_column_width=True)
          else:
              st.markdown('## さすがに、ねこじゃないかもしれない・・・？')
              st.image(annotated_image_pil, caption="Detected Objects.", use_column_width=True)
        else:
            st.markdown('## ねこ以外は、検知する気がありません！')

        
        



if __name__ == "__main__":
    app()
